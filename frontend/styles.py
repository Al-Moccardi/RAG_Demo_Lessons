"""
frontend/styles.py — Custom CSS for the Streamlit App
=======================================================
"""

CUSTOM_CSS = """
<style>
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0d47a1 0%, #6a1b9a 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.1rem; opacity: 0.9; }

    /* Stats cards */
    .stat-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-card h3 { font-size: 2rem; margin: 0; color: #1565c0; }
    .stat-card p { margin: 0.3rem 0 0 0; color: #757575; font-size: 0.9rem; }

    /* Source chunk card */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #1976d2;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }

    /* Pipeline diagram */
    .pipeline-step {
        display: inline-block;
        background: white;
        border: 2px solid #1976d2;
        border-radius: 10px;
        padding: 8px 14px;
        margin: 4px;
        font-size: 0.85rem;
        text-align: center;
    }

    /* Similarity bar */
    .sim-bar {
        display: inline-block;
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #e53935, #fdd835, #43a047);
    }
</style>
"""
