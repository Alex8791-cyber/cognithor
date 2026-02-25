"""Tests für Trusted Circles Ecosystem.

Testet: Circle CRUD, Einladungen, Kuratierung, Reviews,
Trust-Scoring, Collections, Import-Filter.
"""

from __future__ import annotations

import pytest

from jarvis.skills.circles import (
    CircleManager,
    CircleMember,
    CircleRole,
    CuratedCollection,
    CuratedSkill,
    InviteStatus,
    ReviewVerdict,
    SkillReview,
    TrustedCircle,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mgr() -> CircleManager:
    return CircleManager()


@pytest.fixture
def circle(mgr: CircleManager) -> TrustedCircle:
    """Circle mit Owner 'alex'."""
    return mgr.create_circle("Versicherungs-Profis", "alex", owner_name="Alexander")


@pytest.fixture
def full_circle(mgr: CircleManager) -> TrustedCircle:
    """Circle mit 3 Mitgliedern."""
    c = mgr.create_circle("Dev-Team", "alice", owner_name="Alice")
    c.add_member("bob", display_name="Bob", role=CircleRole.ADMIN)
    mgr._peer_circles.setdefault("bob", set()).add(c.circle_id)
    c.add_member("charlie", display_name="Charlie")
    mgr._peer_circles.setdefault("charlie", set()).add(c.circle_id)
    return c


# ============================================================================
# 1. Circle CRUD
# ============================================================================


class TestCircleCRUD:
    def test_create_circle(self, mgr: CircleManager) -> None:
        c = mgr.create_circle("Test", "peer1")
        assert c.name == "Test"
        assert c.member_count == 1
        assert c.owner_id == "peer1"

    def test_create_circle_with_description(self, mgr: CircleManager) -> None:
        c = mgr.create_circle("Pro", "p1", description="Für Profis")
        assert c.description == "Für Profis"

    def test_get_circle(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        found = mgr.get_circle(circle.circle_id)
        assert found is not None
        assert found.name == "Versicherungs-Profis"

    def test_get_nonexistent_circle(self, mgr: CircleManager) -> None:
        assert mgr.get_circle("ghost") is None

    def test_delete_circle_by_owner(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        assert mgr.delete_circle(circle.circle_id, by_peer="alex") is True
        assert mgr.get_circle(circle.circle_id) is None

    def test_delete_circle_by_non_owner_fails(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        assert mgr.delete_circle(circle.circle_id, by_peer="hacker") is False

    def test_list_circles_for_peer(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        circles = mgr.list_circles(peer_id="alex")
        assert len(circles) == 1
        assert circles[0].name == "Versicherungs-Profis"

    def test_list_circles_empty_for_outsider(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        assert mgr.list_circles(peer_id="stranger") == []

    def test_list_all_circles(self, mgr: CircleManager) -> None:
        mgr.create_circle("A", "p1")
        mgr.create_circle("B", "p2")
        assert len(mgr.list_circles()) == 2


# ============================================================================
# 2. Mitglieder-Verwaltung
# ============================================================================


class TestMembers:
    def test_add_member(self, circle: TrustedCircle) -> None:
        m = circle.add_member("bob", display_name="Bob")
        assert m is not None
        assert m.peer_id == "bob"
        assert m.role == CircleRole.MEMBER
        assert circle.member_count == 2

    def test_add_duplicate_fails(self, circle: TrustedCircle) -> None:
        circle.add_member("bob")
        assert circle.add_member("bob") is None

    def test_add_over_limit_fails(self, circle: TrustedCircle) -> None:
        circle.max_members = 2
        circle.add_member("bob")
        assert circle.add_member("charlie") is None

    def test_remove_member(self, circle: TrustedCircle) -> None:
        circle.add_member("bob")
        assert circle.remove_member("bob") is True
        assert not circle.is_member("bob")

    def test_cannot_remove_owner(self, circle: TrustedCircle) -> None:
        assert circle.remove_member("alex") is False

    def test_update_role(self, circle: TrustedCircle) -> None:
        circle.add_member("bob")
        assert circle.update_role("bob", CircleRole.ADMIN) is True
        assert circle.get_member("bob").role == CircleRole.ADMIN

    def test_cannot_demote_owner(self, circle: TrustedCircle) -> None:
        assert circle.update_role("alex", CircleRole.MEMBER) is False

    def test_member_permissions(self) -> None:
        owner = CircleMember(peer_id="o", role=CircleRole.OWNER)
        admin = CircleMember(peer_id="a", role=CircleRole.ADMIN)
        member = CircleMember(peer_id="m", role=CircleRole.MEMBER)
        observer = CircleMember(peer_id="ob", role=CircleRole.OBSERVER)

        assert owner.can_share and owner.can_manage and owner.can_curate
        assert admin.can_share and admin.can_manage and admin.can_curate
        assert member.can_share and not member.can_manage
        assert not observer.can_share and not observer.can_manage


# ============================================================================
# 3. Einladungen
# ============================================================================


class TestInvites:
    def test_create_invite(self, circle: TrustedCircle) -> None:
        invite = circle.create_invite("alex", "bob", message="Komm rein!")
        assert invite is not None
        assert invite.invitee_id == "bob"
        assert invite.status == InviteStatus.PENDING

    def test_non_admin_cannot_invite(self, full_circle: TrustedCircle) -> None:
        assert full_circle.create_invite("charlie", "dave") is None

    def test_accept_invite(self, mgr: CircleManager, circle: TrustedCircle) -> None:
        invite = circle.create_invite("alex", "bob")
        member = mgr.accept_invite(circle.circle_id, invite.invite_id)
        assert member is not None
        assert circle.is_member("bob")
        assert invite.status == InviteStatus.ACCEPTED

    def test_reject_invite(self, circle: TrustedCircle) -> None:
        invite = circle.create_invite("alex", "bob")
        assert circle.reject_invite(invite.invite_id) is True
        assert invite.status == InviteStatus.REJECTED
        assert not circle.is_member("bob")

    def test_cannot_invite_existing_member(self, circle: TrustedCircle) -> None:
        circle.add_member("bob")
        assert circle.create_invite("alex", "bob") is None

    def test_pending_invites(self, circle: TrustedCircle) -> None:
        circle.create_invite("alex", "bob")
        circle.create_invite("alex", "charlie")
        assert len(circle.pending_invites()) == 2

    def test_max_circles_per_peer(self, mgr: CircleManager) -> None:
        mgr.MAX_CIRCLES_PER_PEER = 2
        c1 = mgr.create_circle("C1", "owner1")
        inv1 = c1.create_invite("owner1", "bob")
        mgr.accept_invite(c1.circle_id, inv1.invite_id)

        c2 = mgr.create_circle("C2", "owner2")
        inv2 = c2.create_invite("owner2", "bob")
        mgr.accept_invite(c2.circle_id, inv2.invite_id)

        c3 = mgr.create_circle("C3", "owner3")
        inv3 = c3.create_invite("owner3", "bob")
        assert mgr.accept_invite(c3.circle_id, inv3.invite_id) is None


# ============================================================================
# 4. Kuratierung & Reviews
# ============================================================================


class TestCuration:
    def test_submit_skill(self, full_circle: TrustedCircle) -> None:
        skill = full_circle.submit_skill("pkg1", "BU-Rechner", submitted_by="bob")
        assert skill is not None
        assert skill.name == "BU-Rechner"

    def test_observer_cannot_submit(self, full_circle: TrustedCircle) -> None:
        full_circle.add_member("observer1", role=CircleRole.OBSERVER)
        assert full_circle.submit_skill("pkg", "X", submitted_by="observer1") is None

    def test_review_skill(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="charlie")
        review = full_circle.review_skill("pkg1", "alice", ReviewVerdict.APPROVED, "Gut!")
        assert review is not None
        assert review.verdict == ReviewVerdict.APPROVED

    def test_cannot_self_review(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="bob")
        assert full_circle.review_skill("pkg1", "bob", ReviewVerdict.APPROVED) is None

    def test_member_cannot_review(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="bob")
        assert full_circle.review_skill("pkg1", "charlie", ReviewVerdict.APPROVED) is None

    def test_skill_approved_after_two_reviews(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="charlie")
        full_circle.add_member("admin2", role=CircleRole.ADMIN)
        full_circle.review_skill("pkg1", "alice", ReviewVerdict.APPROVED)
        full_circle.review_skill("pkg1", "bob", ReviewVerdict.APPROVED)

        skill = full_circle.curated_skills["pkg1"]
        assert skill.is_approved is True
        assert skill.approval_count == 2

    def test_rejection_blocks_approval(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="charlie")
        full_circle.review_skill("pkg1", "alice", ReviewVerdict.APPROVED)
        full_circle.review_skill("pkg1", "bob", ReviewVerdict.REJECTED)

        assert full_circle.curated_skills["pkg1"].is_approved is False

    def test_approved_skills_list(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Good", submitted_by="charlie")
        full_circle.submit_skill("pkg2", "Bad", submitted_by="charlie")
        full_circle.review_skill("pkg1", "alice", ReviewVerdict.APPROVED)
        full_circle.review_skill("pkg1", "bob", ReviewVerdict.APPROVED)
        full_circle.review_skill("pkg2", "alice", ReviewVerdict.REJECTED)

        approved = full_circle.approved_skills()
        assert len(approved) == 1
        assert approved[0].name == "Good"

    def test_pending_reviews(self, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "New", submitted_by="charlie")
        assert len(full_circle.pending_reviews()) == 1

    def test_no_review_required(self) -> None:
        c = TrustedCircle(circle_id="x", name="Open", require_review=False)
        c.add_member("alice", role=CircleRole.OWNER)
        c.submit_skill("pkg", "Free", submitted_by="alice")
        assert len(c.approved_skills()) == 1


# ============================================================================
# 5. Trust-Scoring
# ============================================================================


class TestTrustScoring:
    def test_shared_circle_detected(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        assert mgr.is_in_shared_circle("alice", "bob") is True
        assert mgr.is_in_shared_circle("alice", "stranger") is False

    def test_shared_circles_list(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        shared = mgr.shared_circles("alice", "charlie")
        assert len(shared) == 1

    def test_trust_score_with_circle(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        score = mgr.trust_score_for_package("pkg1", "bob", "alice")
        assert score >= CircleManager.CIRCLE_REPUTATION_BOOST

    def test_trust_score_no_circle(self, mgr: CircleManager) -> None:
        score = mgr.trust_score_for_package("pkg1", "stranger1", "stranger2")
        assert score == 0.0

    def test_trust_score_with_curation(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        full_circle.submit_skill("pkg1", "Tool", submitted_by="charlie")
        full_circle.review_skill("pkg1", "alice", ReviewVerdict.APPROVED)
        full_circle.review_skill("pkg1", "bob", ReviewVerdict.APPROVED)

        score = mgr.trust_score_for_package("pkg1", "charlie", "alice")
        assert score >= CircleManager.CIRCLE_REPUTATION_BOOST + 5.0

    def test_filter_trusted_packages(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        publisher_map = {"pkg1": "bob", "pkg2": "stranger", "pkg3": "charlie"}
        result = mgr.filter_trusted_packages(
            ["pkg1", "pkg2", "pkg3"], publisher_map, "alice", min_score=1.0,
        )
        ids = [r[0] for r in result]
        assert "pkg1" in ids
        assert "pkg3" in ids
        assert "pkg2" not in ids

    def test_multiple_circles_boost(self, mgr: CircleManager) -> None:
        c1 = mgr.create_circle("C1", "alice")
        c1.add_member("bob")
        mgr._peer_circles.setdefault("bob", set()).add(c1.circle_id)

        c2 = mgr.create_circle("C2", "alice")
        c2.add_member("bob")
        mgr._peer_circles.setdefault("bob", set()).add(c2.circle_id)

        score = mgr.trust_score_for_package("pkg", "bob", "alice")
        assert score >= 2 * CircleManager.CIRCLE_REPUTATION_BOOST


# ============================================================================
# 6. Kuratierte Sammlungen
# ============================================================================


class TestCollections:
    def test_create_collection(self, mgr: CircleManager) -> None:
        col = mgr.create_collection("Versicherung", "alex", tags=["insurance", "BU"])
        assert col.name == "Versicherung"
        assert "insurance" in col.tags

    def test_add_skill_to_collection(self, mgr: CircleManager) -> None:
        col = mgr.create_collection("Tools", "alex")
        assert col.add_skill("pkg1") is True
        assert col.skill_count == 1

    def test_no_duplicate_in_collection(self, mgr: CircleManager) -> None:
        col = mgr.create_collection("Tools", "alex")
        col.add_skill("pkg1")
        assert col.add_skill("pkg1") is False

    def test_remove_skill_from_collection(self, mgr: CircleManager) -> None:
        col = mgr.create_collection("Tools", "alex")
        col.add_skill("pkg1")
        assert col.remove_skill("pkg1") is True
        assert col.skill_count == 0

    def test_list_collections(self, mgr: CircleManager) -> None:
        mgr.create_collection("A", "p1")
        mgr.create_collection("B", "p2", public=True)
        assert len(mgr.list_collections()) == 2
        assert len(mgr.list_collections(public_only=True)) == 1

    def test_search_collections(self, mgr: CircleManager) -> None:
        mgr.create_collection("BU-Tools", "alex", description="Berufsunfähigkeit", tags=["versicherung"])
        mgr.create_collection("Dev-Tools", "bob", tags=["coding"])
        results = mgr.search_collections("versicherung")
        assert len(results) == 1
        assert results[0].name == "BU-Tools"

    def test_search_by_tag(self, mgr: CircleManager) -> None:
        mgr.create_collection("X", "p", tags=["python", "ai"])
        assert len(mgr.search_collections("python")) == 1


# ============================================================================
# 7. Statistiken
# ============================================================================


class TestStats:
    def test_stats_empty(self, mgr: CircleManager) -> None:
        s = mgr.stats()
        assert s["circles"] == 0
        assert s["total_members"] == 0

    def test_stats_with_data(self, mgr: CircleManager, full_circle: TrustedCircle) -> None:
        mgr.create_collection("Col", "alice")
        s = mgr.stats()
        assert s["circles"] == 1
        assert s["total_members"] == 3
        assert s["collections"] == 1
        assert s["unique_peers"] >= 1
