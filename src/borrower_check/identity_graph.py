"""Bipartite identity linkage graph for synthetic identity ring and velocity detection.

Models the bipartite graph between credit applications (SSN/TIN nodes) and contact/device
identifiers (phone numbers, email addresses, physical address hashes, and device fingerprints).
Computes neighborhood Shannon entropy, sliding-window submission velocity, connected component
cluster sizes, and thin-file Authorized User (AU) tradeline abuse indicators.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class ApplicantIdentity:
    """Normalized identity attributes extracted from an auto-lending credit application."""

    application_id: str
    ssn_hash: str
    phone: str
    email: str
    address_hash: str
    device_fingerprint: str
    timestamp: float
    credit_file_age_months: int = 36
    authorized_user_tradelines: int = 0
    primary_tradelines: int = 3


@dataclass(frozen=True)
class IdentityGraphAnalysis:
    """Risk signals derived from evaluating an applicant against the bipartite identity graph."""

    application_id: str
    shared_ssn_identities: int
    shared_phone_ssns: int
    shared_address_ssns: int
    shared_device_ssns: int
    bipartite_entropy_score: float
    thin_file_au_abuse_flag: bool
    velocity_1h_count: int
    velocity_24h_count: int
    connected_component_size: int
    synthetic_cluster_score: float
    risk_reasons: Tuple[str, ...] = field(default_factory=tuple)


class BipartiteIdentityGraph:
    """In-memory bipartite PII graph for real-time synthetic identity detection."""

    WINDOW_1H_SECONDS: float = 3600.0
    WINDOW_24H_SECONDS: float = 86400.0

    def __init__(self) -> None:
        # Map SSN hash -> set of distinct (phone, email, address_hash) tuples observed
        self._ssn_to_profiles: Dict[str, Set[Tuple[str, str, str]]] = {}
        # Map attribute node (e.g. "phone:+14045550199") -> set of distinct SSN hashes
        self._attr_to_ssns: Dict[str, Set[str]] = {}
        # Map SSN hash -> set of attribute nodes for BFS component traversal
        self._ssn_to_attrs: Dict[str, Set[str]] = {}
        # Map attribute node -> chronological list of application timestamps
        self._attr_timestamps: Dict[str, List[float]] = {}
        # Map SSN node -> chronological list of application timestamps
        self._ssn_timestamps: Dict[str, List[float]] = {}

    @staticmethod
    def _attribute_nodes(identity: ApplicantIdentity) -> Tuple[str, str, str, str]:
        return (
            f"phone:{identity.phone.strip().lower()}",
            f"email:{identity.email.strip().lower()}",
            f"addr:{identity.address_hash.strip().lower()}",
            f"dev:{identity.device_fingerprint.strip().lower()}",
        )

    def _record_velocity(self, key: str, store: Dict[str, List[float]], now: float) -> Tuple[int, int]:
        history = store.setdefault(key, [])
        # Prune timestamps older than 24 hours relative to `now`
        cutoff_24h = now - self.WINDOW_24H_SECONDS
        cutoff_1h = now - self.WINDOW_1H_SECONDS
        pruned = [ts for ts in history if ts >= cutoff_24h]
        pruned.append(now)
        store[key] = pruned
        count_24h = len(pruned)
        count_1h = sum(1 for ts in pruned if ts >= cutoff_1h)
        return count_1h, count_24h

    def _compute_bipartite_entropy(self, ssn_hash: str, attr_nodes: Tuple[str, ...]) -> float:
        """Compute normalized Shannon entropy of SSN-to-PII bipartite degree distribution.

        A clean borrower has a 1-to-1 relationship between their SSN and contact attributes
        (entropy = 0.0). A synthetic identity ring exhibits high fan-out across SSNs and shared
        contact attributes, yielding positive Shannon entropy normalized into [0.0, 1.0].
        """
        degrees: List[int] = []
        ssn_profiles = self._ssn_to_profiles.get(ssn_hash, set())
        if len(ssn_profiles) > 1:
            degrees.append(len(ssn_profiles))
        for attr in attr_nodes:
            linked_ssns = self._attr_to_ssns.get(attr, set())
            if len(linked_ssns) > 1:
                degrees.append(len(linked_ssns))

        if not degrees:
            return 0.0

        total_edges = sum(degrees)
        raw_entropy = 0.0
        for deg in degrees:
            p = deg / total_edges
            raw_entropy -= p * math.log2(p)

        # Combine distribution entropy with edge fan-out magnitude
        magnitude_factor = min(1.0, (total_edges - len(degrees)) / 8.0)
        max_possible_entropy = math.log2(max(2, len(degrees)))
        normalized_entropy = (raw_entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0
        return round(min(1.0, 0.55 * magnitude_factor + 0.45 * normalized_entropy), 4)

    def _connected_component_ssn_count(self, start_ssn: str, max_depth: int = 4) -> int:
        """Traverse bipartite graph via BFS up to `max_depth` hops to size the SSN cluster."""
        visited_ssns: Set[str] = {start_ssn}
        visited_attrs: Set[str] = set()
        queue: deque[Tuple[str, str, int]] = deque([("ssn", start_ssn, 0)])

        while queue:
            node_type, node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            if node_type == "ssn":
                for attr in self._ssn_to_attrs.get(node_id, set()):
                    if attr not in visited_attrs:
                        visited_attrs.add(attr)
                        queue.append(("attr", attr, depth + 1))
            else:
                for linked_ssn in self._attr_to_ssns.get(node_id, set()):
                    if linked_ssn not in visited_ssns:
                        visited_ssns.add(linked_ssn)
                        queue.append(("ssn", linked_ssn, depth + 1))

        return len(visited_ssns)

    def ingest_and_evaluate(self, identity: ApplicantIdentity) -> IdentityGraphAnalysis:
        """Ingest an application identity into the graph and return real-time synthetic fraud signals."""
        ssn = identity.ssn_hash.strip().lower()
        phone_node, email_node, addr_node, dev_node = self._attribute_nodes(identity)
        attr_nodes = (phone_node, email_node, addr_node, dev_node)

        # Update bipartite edges
        profile_tuple = (phone_node, email_node, addr_node)
        self._ssn_to_profiles.setdefault(ssn, set()).add(profile_tuple)
        self._ssn_to_attrs.setdefault(ssn, set()).update(attr_nodes)
        for attr in attr_nodes:
            self._attr_to_ssns.setdefault(attr, set()).add(ssn)

        # Update sliding-window velocity counters across SSN and attributes
        max_1h, max_24h = self._record_velocity(ssn, self._ssn_timestamps, identity.timestamp)
        for attr in attr_nodes:
            c1h, c24h = self._record_velocity(attr, self._attr_timestamps, identity.timestamp)
            max_1h = max(max_1h, c1h)
            max_24h = max(max_24h, c24h)

        shared_ssn_identities = len(self._ssn_to_profiles[ssn])
        shared_phone_ssns = len(self._attr_to_ssns[phone_node])
        shared_address_ssns = len(self._attr_to_ssns[addr_node])
        shared_device_ssns = len(self._attr_to_ssns[dev_node])

        entropy_score = self._compute_bipartite_entropy(ssn, attr_nodes)
        component_size = self._connected_component_ssn_count(ssn)

        # Detect thin-file Authorized User (AU) piggybacking common in synthetic credit boosting
        thin_file_au_abuse = (
            identity.credit_file_age_months < 12
            and identity.authorized_user_tradelines >= 3
            and identity.primary_tradelines <= 1
        )

        reasons: List[str] = []
        if shared_ssn_identities > 1:
            reasons.append(f"SSN_FRAGMENTATION:{shared_ssn_identities}_PROFILES")
        if shared_phone_ssns > 1:
            reasons.append(f"SHARED_PHONE_ACROSS_{shared_phone_ssns}_SSNS")
        if shared_address_ssns > 2:
            reasons.append(f"HIGH_DENSITY_ADDRESS_{shared_address_ssns}_SSNS")
        if shared_device_ssns > 1:
            reasons.append(f"DEVICE_FINGERPRINT_REUSE_{shared_device_ssns}_SSNS")
        if thin_file_au_abuse:
            reasons.append("THIN_FILE_AUTHORIZED_USER_PIGGYBACKING")
        if max_1h >= 3:
            reasons.append(f"VELOCITY_BURST_1H:{max_1h}_APPS")
        elif max_24h >= 5:
            reasons.append(f"VELOCITY_BURST_24H:{max_24h}_APPS")
        if component_size >= 3:
            reasons.append(f"SYNTHETIC_RING_CLUSTER_SIZE_{component_size}")

        # Calculate composite synthetic cluster risk score in [0.0, 1.0]
        raw_score = 0.0
        if shared_ssn_identities > 1:
            raw_score += min(0.35, 0.18 * (shared_ssn_identities - 1))
        if shared_phone_ssns > 1:
            raw_score += min(0.30, 0.16 * (shared_phone_ssns - 1))
        if shared_device_ssns > 1:
            raw_score += min(0.30, 0.18 * (shared_device_ssns - 1))
        if shared_address_ssns > 2:
            raw_score += min(0.20, 0.08 * (shared_address_ssns - 2))
        if thin_file_au_abuse:
            raw_score += 0.28
        if max_1h >= 3:
            raw_score += min(0.25, 0.08 * (max_1h - 2))
        elif max_24h >= 5:
            raw_score += 0.12
        raw_score += 0.20 * entropy_score

        synthetic_cluster_score = round(min(1.0, raw_score), 4)

        return IdentityGraphAnalysis(
            application_id=identity.application_id,
            shared_ssn_identities=shared_ssn_identities,
            shared_phone_ssns=shared_phone_ssns,
            shared_address_ssns=shared_address_ssns,
            shared_device_ssns=shared_device_ssns,
            bipartite_entropy_score=entropy_score,
            thin_file_au_abuse_flag=thin_file_au_abuse,
            velocity_1h_count=max_1h,
            velocity_24h_count=max_24h,
            connected_component_size=component_size,
            synthetic_cluster_score=synthetic_cluster_score,
            risk_reasons=tuple(reasons),
        )
