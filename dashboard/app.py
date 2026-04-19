# ─────────────────────────────────────────────
#  dashboard/app.py
#  Streamlit dashboard — run with:
#  streamlit run dashboard/app.py
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.database import (
    get_recent_events, get_event_counts_by_level,
    get_audit_trail, get_all_embeddings, init_db, get_connection
)
from config.settings import FACES_DIR, DASHBOARD_REFRESH_MS, AlertLevel


# ── Page config ───────────────────────────────
st.set_page_config(
    page_title  = "Smart Security Dashboard",
    page_icon   = "🔒",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
    .alert-red    { background:#ff4444; color:white; padding:8px 14px; border-radius:6px; font-weight:bold; }
    .alert-orange { background:#ff8800; color:white; padding:8px 14px; border-radius:6px; font-weight:bold; }
    .alert-yellow { background:#ffcc00; color:#333;  padding:8px 14px; border-radius:6px; font-weight:bold; }
    .alert-green  { background:#22bb44; color:white; padding:8px 14px; border-radius:6px; font-weight:bold; }
    .metric-box   { background:#1e1e2e; border-radius:10px; padding:16px; text-align:center; }
    .stMetric     { background:#1e1e2e; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)


def level_badge(level: str) -> str:
    badges = {
        "RED":    '<span class="alert-red">🔴 RED</span>',
        "ORANGE": '<span class="alert-orange">🟠 ORANGE</span>',
        "YELLOW": '<span class="alert-yellow">🟡 YELLOW</span>',
        "GREEN":  '<span class="alert-green">🟢 GREEN</span>',
    }
    return badges.get(level, level)


def fmt_ts(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_date(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-checked.png", width=64)
    st.title("Smart Security")
    st.caption("Hierarchical SLM Agent System")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (1s)", value=True)
    filter_level = st.selectbox(
        "Filter by alert level",
        ["ALL", "RED", "ORANGE", "YELLOW", "GREEN"]
    )
    event_limit = st.slider("Events to show", 10, 100, 25)
    st.divider()
    st.caption("🔒 Privacy: No raw faces stored")
    st.caption(f"DB: storage/security.db")

    if st.button("🗑️ Clear all data", type="secondary"):
        st.warning("Are you sure? (restart to confirm)")


# ── Main content ──────────────────────────────
st.title("🔒 Smart Security — Live Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

init_db()

# ── Metric row ────────────────────────────────
counts = get_event_counts_by_level()
total  = sum(counts.values())
faces  = get_all_embeddings()

# Get video processing stats from audit trail
conn = get_connection()
# Count unique videos that completed processing
completed_entries = conn.execute("SELECT details FROM audit_trail WHERE action = 'video_complete'").fetchall()
processed_video_names = set()  # Use set to get unique video names
for entry in completed_entries:
    # Extract video name from "video.mp4 | 0 events | 4.2s"
    video_name = entry[0].split(' | ')[0]
    processed_video_names.add(video_name)

videos_processed = len(processed_video_names)
videos_with_events = conn.execute("SELECT COUNT(DISTINCT source_video) FROM events").fetchone()[0]
conn.close()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Videos Processed", videos_processed)
col2.metric("🔴 Critical (RED)", counts.get("RED", 0))
col3.metric("🟠 Suspicious",     counts.get("ORANGE", 0))
col4.metric("🟡 Repeat Faces",   counts.get("YELLOW", 0))
col5.metric("🟢 Safe Videos",    videos_processed - videos_with_events)
col6.metric("👤 Known Faces",    len(faces))

# ── Status indicator ──────────────────────────
if videos_processed > 0:
    safe_videos = videos_processed - videos_with_events
    if safe_videos == videos_processed:
        st.success(f"✅ All {videos_processed} processed videos are SAFE (GREEN status)")
    else:
        st.info(f"📊 {safe_videos}/{videos_processed} videos processed safely")

    # Show which videos are safe
    if safe_videos > 0:
        conn = get_connection()
        event_videos = conn.execute("SELECT DISTINCT source_video FROM events").fetchall()
        event_video_names = [row[0] for row in event_videos]
        safe_video_list = list(processed_video_names - set(event_video_names))  # Convert set difference to list
        conn.close()

        st.subheader("🟢 Safe Videos")
        for video in sorted(safe_video_list):
            st.write(f"✅ {video}")
else:
    st.info("🎥 No videos processed yet. Run: `python main.py --video <path>`")

st.divider()

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Live Events", "📊 Analytics", "👤 Face Re-ID", "📜 Audit Trail"
])

# ══ Tab 1: Live Events ════════════════════════
with tab1:
    level_filter = None if filter_level == "ALL" else filter_level
    events = get_recent_events(limit=event_limit, level=level_filter)

    if not events:
        if videos_processed > 0:
            st.success(f"✅ {videos_processed} video(s) processed successfully - All safe! (GREEN status)")
        else:
            st.info("No events yet. Start the pipeline: `python main.py --video <path>`")
    else:
        for ev in events:
            level = ev.get("alert_level", "GREEN")
            etype = ev.get("event_type", "unknown")
            ts    = fmt_ts(ev.get("timestamp", 0))
            desc  = ev.get("description", ev.get("manager_verdict", "No description."))
            tid   = ev.get("track_id", "?")
            zone  = ev.get("zone_id", "—")
            src   = ev.get("source_video", "—")

            with st.expander(
                f"{ts}  |  {etype.upper()}  |  Zone: {zone}  |  Track: {tid}",
                expanded=(level in ("RED", "ORANGE"))
            ):
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(level_badge(level), unsafe_allow_html=True)
                    st.caption(f"Track ID: {tid}")
                    if ev.get("face_hash"):
                        st.caption(f"Face: {ev['face_hash'][:8]}…")
                        # Show blurred face crop if exists
                        face_img = Path(FACES_DIR) / f"{ev['face_hash']}.jpg"
                        if face_img.exists():
                            st.image(str(face_img), caption="Blurred face (privacy)", width=80)
                with col_b:
                    st.write(desc or "—")
                    if ev.get("manager_verdict"):
                        try:
                            verdict = json.loads(ev["manager_verdict"]) if isinstance(ev["manager_verdict"], str) else ev["manager_verdict"]
                            st.caption(f"Verdict: {verdict.get('reasoning','—')}")
                        except Exception:
                            st.caption(ev["manager_verdict"])
                    st.caption(f"Source: {src}  |  {fmt_date(ev.get('timestamp'))}")


# ══ Tab 2: Analytics ══════════════════════════
with tab2:
    all_events = get_recent_events(limit=500)

    if not all_events:
        st.info("No data yet.")
    else:
        df = pd.DataFrame(all_events)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"]     = df["datetime"].dt.floor("h")

        col_l, col_r = st.columns(2)

        # Alert level pie
        with col_l:
            st.subheader("Alert Level Distribution")
            if counts:
                fig = px.pie(
                    names  = list(counts.keys()),
                    values = list(counts.values()),
                    color  = list(counts.keys()),
                    color_discrete_map = {
                        "RED": "#ff4444", "ORANGE": "#ff8800",
                        "YELLOW": "#ffcc00", "GREEN": "#22bb44"
                    },
                    hole   = 0.4,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

        # Event type bar chart
        with col_r:
            st.subheader("Events by Type")
            type_counts = df["event_type"].value_counts().reset_index()
            type_counts.columns = ["event_type", "count"]
            fig2 = px.bar(
                type_counts, x="event_type", y="count",
                color="count", color_continuous_scale="Reds"
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

        # Timeline
        st.subheader("Event Timeline")
        if "hour" in df.columns:
            timeline = df.groupby(["hour", "alert_level"]).size().reset_index(name="count")
            fig3 = px.bar(
                timeline, x="hour", y="count", color="alert_level",
                color_discrete_map={
                    "RED": "#ff4444", "ORANGE": "#ff8800",
                    "YELLOW": "#ffcc00", "GREEN": "#22bb44"
                },
                barmode="stack"
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig3, use_container_width=True)


# ══ Tab 3: Face Re-ID ═════════════════════════
with tab3:
    st.subheader("👤 Face Re-ID Database")
    st.caption("🔒 Only anonymised embeddings stored — no raw face images in database")

    face_records = get_all_embeddings()

    if not face_records:
        st.info("No faces registered yet.")
    else:
        face_df = pd.DataFrame([
            {
                "Face Hash":    r["face_hash"],
                "Seen Count":   r["seen_count"],
                "First Seen":   fmt_date(r["first_seen"]),
                "Alert Issued": "✅ Yes" if r["alert_issued"] else "No",
                "Status":       "🟡 REPEAT" if r["seen_count"] > 1 else "🆕 New",
            }
            for r in face_records
        ])

        # Highlight repeat faces
        st.dataframe(
            face_df.sort_values("Seen Count", ascending=False),
            use_container_width=True, hide_index=True
        )

        repeats = sum(1 for r in face_records if r["seen_count"] > 1)
        st.metric("Repeat individuals", repeats, delta=f"of {len(face_records)} total")

        # Show blurred face crops
        st.subheader("Logged Face Crops (blurred)")
        crops = list(Path(FACES_DIR).glob("*.jpg"))
        if crops:
            cols = st.columns(min(len(crops), 6))
            for i, crop_path in enumerate(crops[:12]):
                with cols[i % 6]:
                    st.image(str(crop_path), caption=crop_path.stem[:8], width=80)
        else:
            st.caption("No blurred crops saved yet.")


# ══ Tab 4: Audit Trail ════════════════════════
with tab4:
    st.subheader("📜 System Audit Trail")
    trail = get_audit_trail(limit=100)
    if not trail:
        st.info("No audit entries yet.")
    else:
        trail_df = pd.DataFrame([
            {
                "Time":    fmt_date(r["timestamp"]),
                "Action":  r["action"],
                "Details": r["details"],
                "Source":  r["source"],
            }
            for r in trail
        ])
        st.dataframe(trail_df, use_container_width=True, hide_index=True)


# ── Auto-refresh ──────────────────────────────
if auto_refresh:
    time.sleep(1)
    st.rerun()
