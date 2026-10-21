//! The control flow graph of a body as the borrow checker walks it: every edge, the imaginary
//! ones of `FalseEdge` and `FalseUnwind` included, and a number for each location.

use la_arena::RawIdx;
use smallvec::SmallVec;

use hir_ty::mir::{BasicBlockId, MirBody, TerminatorKind};

/// A statement of a block, or its terminator when `statement` is the number of statements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(super) struct Location {
    pub(super) block: BasicBlockId,
    pub(super) statement: usize,
}

pub(super) struct Cfg {
    /// The number of the first location of each block, by its index.
    starts: Vec<usize>,
    pub(super) points: usize,
    pub(super) successors: Vec<SmallVec<[BasicBlockId; 2]>>,
    pub(super) predecessors: Vec<SmallVec<[BasicBlockId; 4]>>,
    /// The blocks reachable from the start, in reverse postorder.
    pub(super) order: Vec<BasicBlockId>,
    pub(super) reachable: Vec<bool>,
}

pub(super) fn index(block: BasicBlockId) -> usize {
    block.into_raw().into_u32() as usize
}

pub(super) fn block(index: usize) -> BasicBlockId {
    BasicBlockId::from_raw(RawIdx::from_u32(index as u32))
}

/// Where control may go after `kind`, unwinding and the edges that never run included.
pub(super) fn successors(kind: &TerminatorKind) -> SmallVec<[BasicBlockId; 2]> {
    match kind {
        TerminatorKind::FalseEdge { real_target, imaginary_target } => {
            smallvec::smallvec![*real_target, *imaginary_target]
        }
        _ => {
            let mut all = kind.successors();
            all.extend(kind.unwind());
            all
        }
    }
}

impl Cfg {
    pub(super) fn new(body: &MirBody<'_>) -> Self {
        let count = body.basic_blocks.len();
        let mut starts = Vec::with_capacity(count);
        let mut points = 0;
        let mut successors = vec![SmallVec::new(); count];
        let mut predecessors = vec![SmallVec::new(); count];
        for (id, data) in body.basic_blocks.iter() {
            starts.push(points);
            points += data.statements.len() + 1;
            if let Some(terminator) = &data.terminator {
                for target in self::successors(&terminator.kind) {
                    if !successors[index(id)].contains(&target) {
                        successors[index(id)].push(target);
                        predecessors[index(target)].push(id);
                    }
                }
            }
        }
        for &(from, _, cleanup) in &body.panic_unwinds {
            if !successors[index(from)].contains(&cleanup) {
                successors[index(from)].push(cleanup);
                predecessors[index(cleanup)].push(from);
            }
        }
        let mut reachable = vec![false; count];
        let mut postorder = Vec::with_capacity(count);
        let mut stack = vec![(body.start_block, 0usize)];
        reachable[index(body.start_block)] = true;
        while let Some((block, next)) = stack.last_mut() {
            let block = *block;
            if let Some(&target) = successors[index(block)].get(*next) {
                *next += 1;
                if !reachable[index(target)] {
                    reachable[index(target)] = true;
                    stack.push((target, 0));
                }
            } else {
                postorder.push(block);
                stack.pop();
            }
        }
        postorder.reverse();
        Cfg { starts, points, successors, predecessors, order: postorder, reachable }
    }

    pub(super) fn point(&self, location: Location) -> usize {
        self.starts[index(location.block)] + location.statement
    }

    pub(super) fn location(&self, point: usize) -> Location {
        let block = self.starts.partition_point(|&start| start <= point) - 1;
        Location { block: self::block(block), statement: point - self.starts[block] }
    }

    pub(super) fn block_start(&self, block: BasicBlockId) -> usize {
        self.starts[index(block)]
    }

    /// The last location of `block`, its terminator.
    pub(super) fn block_end(&self, body: &MirBody<'_>, block: BasicBlockId) -> usize {
        self.starts[index(block)] + body.basic_blocks[block].statements.len()
    }

    pub(super) fn is_reachable(&self, block: BasicBlockId) -> bool {
        self.reachable[index(block)]
    }

    /// The locations control may come to `location` from.
    pub(super) fn predecessor_locations(
        &self,
        body: &MirBody<'_>,
        location: Location,
    ) -> SmallVec<[Location; 4]> {
        if location.statement > 0 {
            return smallvec::smallvec![Location { statement: location.statement - 1, ..location }];
        }
        self.predecessors[index(location.block)]
            .iter()
            .filter(|it| self.is_reachable(**it))
            .map(|&block| Location { block, statement: body.basic_blocks[block].statements.len() })
            .collect()
    }
}

/// The immediate dominator of each reachable block (Cooper, Harvey and Kennedy).
pub(super) struct Dominators {
    idom: Vec<usize>,
    rank: Vec<usize>,
}

impl Dominators {
    pub(super) fn new(cfg: &Cfg) -> Self {
        let count = cfg.reachable.len();
        let mut rank = vec![usize::MAX; count];
        for (position, &block) in cfg.order.iter().enumerate() {
            rank[index(block)] = position;
        }
        let mut idom = vec![usize::MAX; count];
        let Some(&start) = cfg.order.first() else { return Dominators { idom, rank } };
        idom[index(start)] = index(start);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.order[1..] {
                let mut new = usize::MAX;
                for &pred in &cfg.predecessors[index(block)] {
                    let pred = index(pred);
                    if idom[pred] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        pred
                    } else {
                        let (mut a, mut b) = (pred, new);
                        while a != b {
                            while rank[a] > rank[b] {
                                a = idom[a];
                            }
                            while rank[b] > rank[a] {
                                b = idom[b];
                            }
                        }
                        a
                    };
                }
                if new != usize::MAX && idom[index(block)] != new {
                    idom[index(block)] = new;
                    changed = true;
                }
            }
        }
        Dominators { idom, rank }
    }

    pub(super) fn dominates(&self, a: BasicBlockId, b: BasicBlockId) -> bool {
        let (a, mut b) = (index(a), index(b));
        if self.rank[b] == usize::MAX {
            return false;
        }
        loop {
            if a == b {
                return true;
            }
            let up = self.idom[b];
            if up == b || up == usize::MAX {
                return false;
            }
            b = up;
        }
    }

    pub(super) fn location_dominates(&self, a: Location, b: Location) -> bool {
        if a.block == b.block {
            a.statement <= b.statement
        } else {
            self.dominates(a.block, b.block)
        }
    }
}
