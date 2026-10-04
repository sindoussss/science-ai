"""Invalidation cascade. Runs in SQLite so it reaches every session, not just the open one."""
from __future__ import annotations

from sciai.graph.model import CascadeReport, NodeType, Status
from sciai.store.repository import Repository


def cascade_invalidate(repo: Repository, origin_id: str, reason: str,
                       current_session: str | None) -> CascadeReport:
    """Invalidate every transitive dependent of ``origin_id``.

    The origin's own status is set by the caller. Locked dependents are still
    invalidated (the status must stay truthful) but are reported as warnings.
    Other sessions that contained an affected node get a persistent notice.
    """
    report = CascadeReport(origin_id=origin_id)
    dependents = repo.dependents_transitive(origin_id)
    nodes = repo.get_nodes(dependents)
    with repo.db.tx():
        for node in nodes:
            if node.status == Status.INVALIDATED:
                continue
            old = node.status
            node.status = Status.INVALIDATED
            repo.update_node(node)
            repo.log_status(node.id, old.value, node.status.value, f"cascade: {reason}", origin_id)
            report.invalidated.append(node.id)
            if node.locked:
                report.locked_warnings.append(node.id)

        touched = [origin_id, *report.invalidated]
        by_session = repo.sessions_containing(touched)
        finals = {n.id: n for n in nodes if n.type == NodeType.FINAL}
        origin = repo.get_node(origin_id)
        if origin is not None and origin.type == NodeType.FINAL:
            finals[origin.id] = origin
        for sid, ids in by_session.items():
            if sid == current_session:
                continue
            answers = [(i, finals[i].title) for i in ids if i in finals]
            report.affected_sessions[sid] = answers
            if answers:
                text = (f"{len(answers)} answer(s) in this session were invalidated because a result "
                        f"they depend on failed: {reason}")
            else:
                text = f"{len(ids)} node(s) in this session were invalidated: {reason}"
            repo.add_notice(sid, origin_id, text,
                            {"invalidated": ids, "answers": answers, "reason": reason})
    return report
