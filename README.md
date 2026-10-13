<<<<<<< HEAD
# 🏥 MediFlow AI — Hospital Resource & Flow Optimization

> **Hackathon Track:** Overcrowded Hospitals, Limited Resources: Doing More With What Exists  
> **Phase:** Phase 2 — Hospital What-If Simulator Active

---

## 📌 Problem Statement
Hospitals routinely face sudden surges in patient arrivals, bed bottlenecks, diagnostic equipment outages (like CT/MRI scanners), and discharge delays. Staff are stretched thin, and hospital administrators often lack clear, real-time visibility into how resources are flowing across departments.

**MediFlow AI** is designed to help hospitals understand resource usage, detect bottlenecks, and explore ways to improve operational flow—doing more with existing capacity without compromising patient care.

---

## 🚀 Features Implemented

### Phase 1: Hospital Resource Dashboard
- **Key Operational Metrics:** Real-time visibility into bed occupancy, available beds, patient waiting queues, and staffing levels.
- **Dynamic Arrival Timeline:** 24-hour visual trend of patient arrivals segmented by severity (Emergency, Urgent, Routine).
- **Departmental Bed Utilization:** Breakdown of bed capacity across ICU, Emergency, General Ward, Surgery, and Pediatrics.
- **Resource & Equipment Status Tables:** Overview of unit capacities, staffing allocations, and diagnostic equipment readiness.

### Phase 2: Hospital What-If Simulator (NEW)
- **Interactive Disruption Controls:**
  - Additional patient arrivals slider (0 to 60 patients).
  - CT Scanner status selector (Available vs. Unavailable 50% outage).
  - Inpatient discharge delay toggle (On vs. Off).
  - Simulation duration selector (4, 8, 12, or 24 hours).
- **Three Side-by-Side Scenarios:**
  - **Scenario A (Baseline):** Standard hospital operations without disruptions.
  - **Scenario B (Disrupted):** Selected stress-test disruption showing compounding strain on beds and diagnostics.
  - **Scenario C (Proposed Response):** Illustrative mitigation strategy (e.g. surge flex beds, dedicated discharge lounge, and diagnostic diversion).
- **Consistent Resource Flow Math:** Hard capacity caps (beds occupied cannot exceed physical capacity, available beds $\ge 0$, and excess demand is accurately tracked in waiting queues).
- **Interactive Visualizations:**
  - Bed allocation bar chart comparing Occupied, Available, and Unmet Waiting Demand.
  - Hour-by-hour patient waiting queue trajectory.
  - Diagnostic CT scanner queue lengths and estimated wait times.
- **Primary Bottleneck Detection:** Clear identification of dominant operational constraints (Inpatient Bed Saturation vs. Diagnostic Backlog vs. Discharge Turnover Delays).

---

## 🛠️ Technology Stack
- **Python 3.10+** (tested on Python 3.13)
- **[Streamlit](https://streamlit.io/):** Interactive web application framework for rapid prototyping
- **[Pandas](https://pandas.pydata.org/):** Data structuring and deterministic resource flow calculations
- **[Plotly](https://plotly.com/python/):** Interactive, polished charts and data visualisations

---

## 💻 Quick Start Guide for Beginners

### 1. Prerequisites
Ensure you have Python installed on your computer. You can check by running:
```bash
python --version
```

### 2. Navigate to Project Folder
Open your terminal (PowerShell, Command Prompt, or VS Code Terminal) in the project directory:
```bash
cd "c:\Users\admin\Documents\Projects\SRM"
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the MediFlow AI dashboard with:
```bash
streamlit run app.py
```

Streamlit will automatically open the application in your browser (usually at `http://localhost:8501`).

---

## 📂 Project Structure
```text
SRM/
│
├── app.py              # Main Streamlit web application & UI controllers
├── simulator.py        # Modular Hospital What-If resource simulation engine
├── requirements.txt    # List of required Python packages
└── README.md           # Setup guide and project documentation
```

---

## 🧭 Roadmap & Upcoming Phases
- [x] **Phase 1:** Project setup, synthetic hospital data engine, and resource dashboard
- [x] **Phase 2:** Hospital What-If Simulator (Surge arrivals, CT outages, discharge delays & scenario comparison)
- [ ] **Phase 3:** Bottleneck Detection & Automated Recommendation Engine
- [ ] **Phase 4:** Basic Patient Arrival Forecasting & Hackathon Presentation Polish

---

## ⚠️ Disclaimer
All data displayed in MediFlow AI is **synthetically generated demonstration data**. No real patient information or proprietary hospital systems are used or accessed.
=======
# mediflow-ai
>>>>>>> 17b98041841525a235908464e1f306bf524b39fb
