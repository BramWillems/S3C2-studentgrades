import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ----------------------------------------
st.set_page_config(
    page_title="Knowledge Tracing Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Knowledge Tracing Analytics Dashboard")
st.markdown("Exploring Population-Level Skill Parameters (BKT) and Dynamic Student Mastery (DKT)")

# ----------------------------------------
# 2. DATA LOADING (CACHED FOR SPEED)
# ----------------------------------------
@st.cache_data
def load_data():
    # Replace these with your actual computed dataframes/paths
    try:
        bkt_df = pd.read_csv("data/bkt_parameters.csv", index_col=0)
    except FileNotFoundError:
        # Mock data for demonstration if files don't exist yet
        skills = [f"Skill_{i}" for i in range(1, 51)]
        bkt_df = pd.DataFrame({
            'skill_name': skills,
            'Prior': np.random.uniform(0.1, 0.6, 50),
            'Learn': np.random.uniform(0.05, 0.3, 50),
            'Guess': np.random.uniform(0.1, 0.3, 50),
            'Slip': np.random.uniform(0.05, 0.2, 50),
            'n_obs': np.random.randint(10, 500, 50)
        }).set_index('skill_name')
        
    return bkt_df

bkt_df = load_data()

# ----------------------------------------
# 3. NAVIGATION TABS
# ----------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Global Skill Map (Macro)", 
    "🎯 Pop-Level Skill Explorer (BKT)", 
    "🏃 Student Journey Simulator (DKT)"
])

# ========================================
# TAB 1: GLOBAL SKILL MAP
# ========================================
with tab1:
    st.header("Global Overview of All Skills")
    st.markdown("Analyze how skills compare across difficulty (`Prior`) and learnability (`Learn Rate`).")
    
    # Reliability Filter Setup
    st.sidebar.header("Global Filters")
    min_obs = st.sidebar.slider("Minimum observations (n_obs) for reliability", 0, 300, 50)
    
    filtered_df = bkt_df[bkt_df['n_obs'] >= min_obs]
    st.metric("Analyzing Reliable Skills", f"{len(filtered_df)} / {len(bkt_df)}")
    
    # Interactive Scatter Plot
    fig_scatter = px.scatter(
        filtered_df, 
        x="Prior", 
        y="Learn", 
        size="n_obs",
        hover_name=filtered_df.index,
        labels={"Prior": "Initial Difficulty (Prior)", "Learn": "Transition/Learn Rate"},
        title="Skill Map: Difficulty vs. Learnability",
        color="Learn",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========================================
# TAB 2: POP-LEVEL SKILL EXPLORER
# ========================================
with tab2:
    st.header("Bayesian Knowledge Tracing Skill Inspector")
    
    # Dropdown selection for skills
    selected_skill = st.selectbox("Select a skill to inspect:", bkt_df.index)
    skill_data = bkt_df.loc[selected_skill]
    
    # Key Metrics Cards Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Prior (Initial Knowledge)", value=f"{skill_data['Prior']:.3f}")
    col2.metric(label="Learn Rate (Grasp Speed)", value=f"{skill_data['Learn']:.3f}")
    col3.metric(label="Guess Probability", value=f"{skill_data['Guess']:.3f}")
    col4.metric(label="Slip Probability", value=f"{skill_data['Slip']:.3f}")
    
    # Simulated Empirical Learning Curve based on BKT parameters
    st.subheader("Theoretical Learning Curve Progression")
    t = np.arange(1, 11)
    # Standard forward BKT updating sequence equations for mastery P(L_t)
    p_lost = []
    p_l = skill_data['Prior']
    for step in t:
        p_lost.append(p_l)
        # Assuming no feedback scenario or general forward projection:
        p_l = p_l + (1 - p_l) * skill_data['Learn']
        
    p_correct_curve = [pl * (1 - skill_data['Slip']) + (1 - pl) * skill_data['Guess'] for pl in p_lost]
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=t, y=p_correct_curve, mode='lines+markers', name='P(Correct)', line=dict(color='#1D9E75', width=3)))
    fig_curve.update_layout(
        title=f"Expected Performance Growth over Sequential Opportunities for '{selected_skill}'",
        xaxis_title="Opportunity Number",
        yaxis_title="Probability of Correct Answer",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# ========================================
# TAB 3: STUDENT JOURNEY SIMULATOR
# ========================================
with tab3:
    st.header("Deep Knowledge Tracing Student Walkthrough")
    st.markdown("Visualize an individual student's path and how a dynamic recurrent neural network maps performance updates.")
    
    # Hardcoded or simulated sequence for presentation visualization
    student_id = st.text_input("Enter Student ID to inspect:", "User_40291")
    
    st.write(f"Displaying dynamic tracking sequence for **{student_id}** across interaction timeline:")
    
    # Create fake timeline sequence representing DKT tracker
    steps = list(range(1, 16))
    mock_probabilities = [0.4, 0.42, 0.39, 0.65, 0.68, 0.62, 0.81, 0.85, 0.83, 0.91, 0.94, 0.92, 0.96, 0.97, 0.98]
    mock_correctness = [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    mock_skills = ["Addition", "Addition", "Addition", "Addition", "Subtraction", "Subtraction", "Subtraction", "Multiplication", "Multiplication", "Multiplication", "Division", "Division", "Division", "Division", "Division"]
    
    fig_dkt = go.Figure()
    
    # Line chart representing model's hidden estimate state
    fig_dkt.add_trace(go.Scatter(
        x=steps, y=mock_probabilities,
        mode='lines', name='DKT P(Correct) Projection',
        line=dict(color='#A3E4D7', width=2, dash='dot')
    ))
    
    # Green markers for success
    correct_indices = [i for i, x in enumerate(mock_correctness) if x == 1]
    fig_dkt.add_trace(go.Scatter(
        x=[steps[i] for i in correct_indices],
        y=[mock_probabilities[i] for i in correct_indices],
        mode='markers', name='Correct Response (✓)',
        marker=dict(color='#1D9E75', size=14, symbol='triangle-up'),
        text=[mock_skills[i] for i in correct_indices]
    ))
    
    # Orange markers for errors
    incorrect_indices = [i for i, x in enumerate(mock_correctness) if x == 0]
    fig_dkt.add_trace(go.Scatter(
        x=[steps[i] for i in incorrect_indices],
        y=[mock_probabilities[i] for i in incorrect_indices],
        mode='markers', name='Incorrect Response (✗)',
        marker=dict(color='#EF9F27', size=14, symbol='triangle-down'),
        text=[mock_skills[i] for i in incorrect_indices]
    ))
    
    fig_dkt.update_layout(
        xaxis_title="Interaction Order Number (Across System)",
        yaxis_title="Model Latent Knowledge Estimate",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_dkt, use_container_width=True)