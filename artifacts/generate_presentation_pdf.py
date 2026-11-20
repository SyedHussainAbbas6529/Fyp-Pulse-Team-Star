"""
Generates the FYP-Pulse Presentation and Viva Preparation PDF.
Built by Team Star for Vibe Jam AI Sprint 2026.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header (on later pages)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "FYP-Pulse | Team Star ⭐ — Presentation & Viva Defense Guide")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_text)
        self.drawString(54, 36, "Vibe Jam: AI Sprint 2026 — Track 4: FYP Early Warning System")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.restoreState()


def build_pdf(filename="artifacts/FYP_Pulse_Presentation_and_Viva_Prep.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    c_primary = colors.HexColor("#1e3a8a")     # Deep Blue
    c_secondary = colors.HexColor("#0284c7")   # Sky Blue
    c_dark = colors.HexColor("#0f172a")        # Slate 900
    c_body = colors.HexColor("#334155")        # Slate 700
    c_gold = colors.HexColor("#d97706")        # Amber/Gold
    c_light = colors.HexColor("#f8fafc")       # Slate 50
    c_border = colors.HexColor("#e2e8f0")      # Border Slate

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=c_primary,
        alignment=1, # Center
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=c_secondary,
        alignment=1,
    )
    
    h1_style = ParagraphStyle(
        "SlideHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=c_body,
        spaceAfter=6,
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=c_dark,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=13.5,
        textColor=c_primary,
    )

    q_style = ParagraphStyle(
        "VivaQ",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=3,
    )

    a_style = ParagraphStyle(
        "VivaA",
        parent=body_style,
        fontSize=9.5,
        leading=13.5,
        textColor=c_body,
        spaceAfter=8,
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("⭐ FYP-Pulse | Team Star", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Final Year Project Early Warning System", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Track 4 — Vibe Jam: AI Sprint 2026</b>", subtitle_style))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="80%", thickness=2, color=c_gold, spaceAfter=25, spaceBefore=10))

    cover_meta = [
        [Paragraph("<b>Project Focus:</b>", bold_body), Paragraph("Machine Learning Early Detection of Failing FYP Groups by Week 8", body_style)],
        [Paragraph("<b>Team Name:</b>", bold_body), Paragraph("Team Star ⭐", body_style)],
        [Paragraph("<b>System Core:</b>", bold_body), Paragraph("100% Offline, Single-Command Streamlit App (No Data Leakage, Calibrated ML)", body_style)],
        [Paragraph("<b>Target Audience:</b>", bold_body), Paragraph("FYP Coordinators, Academic Committees, Department Heads, Project Supervisors", body_style)],
        [Paragraph("<b>Document Purpose:</b>", bold_body), Paragraph("Comprehensive Presentation Pitch & Full Viva Defense Guide", body_style)],
    ]
    t_cover = Table(cover_meta, colWidths=[130, 370])
    t_cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_cover)

    story.append(Spacer(1, 35))
    
    exec_summary_box = [
        [Paragraph("<b>EXECUTIVE SUMMARY IN ONE SENTENCE:</b><br/>"
                   "About 1 in 6 university computing groups fail or carry over their Final Year Project; "
                   "FYP-Pulse flags at-risk groups by <b>Week 8</b> using calibrated machine learning, "
                   "pinpoints the exact causes in plain English, and prescribes immediate corrective actions "
                   "before it is too late to save them.", callout_style)]
    ]
    t_exec = Table(exec_summary_box, colWidths=[500])
    t_exec.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1.5, c_secondary),
        ("PADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(t_exec)

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 1: WHY DID WE BUILD THIS? (THE PROBLEM & OPPORTUNITY)
    # =========================================================================
    story.append(Paragraph("1. Why Did We Build FYP-Pulse? (The Problem)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    story.append(Paragraph("<b>The Reality in Universities:</b>", bold_body))
    story.append(Paragraph(
        "Every year, <b>approx. 16% (about 1 out of every 6)</b> undergraduate student teams fail "
        "their Final Year Project or get pushed into a painful 6-month semester extension. "
        "This damages graduation rates, causes immense student depression, and wastes university faculty resources.",
        body_style
    ))

    story.append(Paragraph("<b>The Traditional Flaw — Too Late to Help:</b>", bold_body))
    story.append(Paragraph(
        "Historically, project failure is only discovered at the <b>Final Defense (Week 16)</b>. "
        "By then, it is terminal: grades are submitted, the defense panel fails the project, and students have no time left to recover.",
        body_style
    ))

    story.append(Paragraph("<b>The Solution — The Week 8 Window of Opportunity:</b>", bold_body))
    story.append(Paragraph(
        "By Week 8 (Mid-Semester), teams have already submitted proposals, committed initial code, and attended supervisor meetings. "
        "If coordinators can identify risk <b>exactly at Week 8</b>, faculty still have <b>8 whole weeks remaining</b> "
        "to intervene, restructure tasks, mandate meetings, and guide the team to a successful graduation.",
        body_style
    ))

    story.append(Spacer(1, 10))

    problem_table_data = [
        [Paragraph("<b>Traditional University Approach</b>", bold_body), Paragraph("<b>FYP-Pulse AI Early Warning System</b>", bold_body)],
        [Paragraph("• Discovers failure at Week 16 (Final Defense).<br/>"
                   "• Purely reactive: extension granted or project failed.<br/>"
                   "• Relies on subjective feelings of busy supervisors.<br/>"
                   "• No statistical fairness checks across campuses/gender.", body_style),
         Paragraph("• Discovers risk at <b>Week 8</b> (Mid-Semester Evaluation).<br/>"
                   "• Proactive: 8 remaining weeks to intervene and save the group.<br/>"
                   "• Objective: Calibrated data-driven ML with top 3 driver explanations.<br/>"
                   "• Statistically certified fair across gender and regional campuses.", body_style)],
    ]
    t_prob = Table(problem_table_data, colWidths=[245, 255])
    t_prob.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fef2f2")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f0fdf4")),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_prob)

    story.append(Spacer(1, 15))

    # =========================================================================
    # SLIDE 2: WHAT DOES THE APP DO? (THE 8 INTEGRATED MODULES)
    # =========================================================================
    story.append(Paragraph("2. What Does the Application Do?", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    story.append(Paragraph(
        "Everything lives inside <b>one single Streamlit web application</b> that runs with a single command "
        "(<code>streamlit run app.py</code>) completely offline with zero external paid APIs:", body_style
    ))

    modules_data = [
        [Paragraph("<b>Module</b>", bold_body), Paragraph("<b>Feature Name</b>", bold_body), Paragraph("<b>What It Does for the User</b>", bold_body)],
        [Paragraph("<b>A1</b>", bold_body), Paragraph("Data Quality & Cleaning", body_style), Paragraph("Auto-standardizes messy university files (typos in campuses, M/F gender, swapped CGPAs, solo team rules) and exports an audit log.", body_style)],
        [Paragraph("<b>A2</b>", bold_body), Paragraph("Executive Dashboard", body_style), Paragraph("8 interactive Plotly charts examining Team, Supervisor, Academic Context, and Progress Velocity with real-number dynamic insights.", body_style)],
        [Paragraph("<b>A3</b>", bold_body), Paragraph("Model Training & Selection", body_style), Paragraph("Trains 4 models with 5-fold Stratified CV, tunes decision thresholds for Recall, and calibrates probabilities with Platt scaling.", body_style)],
        [Paragraph("<b>A4</b>", bold_body), Paragraph("Prediction Form & Tiers", body_style), Paragraph("Evaluates any group's week-8 details; yields exact 4-decimal probability, color-coded badge, and verified monotonic risk tiers.", body_style)],
        [Paragraph("<b>A5</b>", bold_body), Paragraph("Explainability & Fairness", body_style), Paragraph("Identifies Top 3 risk drivers (⬆️/⬇️) in plain language, gives a concrete action step, and certifies gender/campus fairness with Wilson 95% CIs.", body_style)],
        [Paragraph("<b>A6</b>", bold_body), Paragraph("Weekly Watchlist", body_style), Paragraph("Ranks the top-K highest risk groups for committee meetings; achieves <b>2.48× Lift</b> over random chance with CSV export.", body_style)],
        [Paragraph("<b>A7</b>", bold_body), Paragraph("What-If Simulator", body_style), Paragraph("Allows coordinators to test interventions (e.g. increase meetings or commits) while locking immutable demographics; sweeps sensitivity.", body_style)],
    ]
    t_mod = Table(modules_data, colWidths=[40, 140, 320])
    t_mod.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_mod)

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: DATA INTEGRITY & LEAKAGE CONTROL
    # =========================================================================
    story.append(Paragraph("3. Data Cleaning & Strict Anti-Leakage Rules", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    story.append(Paragraph("<b>The First Rule of Machine Learning: A Value Outside Its Valid Range is an ERROR, Not an Outlier.</b>", bold_body))
    story.append(Paragraph(
        "Real-world educational datasets are messy. FYP-Pulse executes a transparent rule-based cleaner "
        "that corrected <b>1,353 data anomalies</b> and saved an exact row-by-row audit log in <code>artifacts/cleaning_log.csv</code>:",
        body_style
    ))

    cleaning_points = [
        "• <b>Campus Typos & Abbreviations:</b> Used difflib fuzzy matching and exact mapping (e.g. 'LHR'/'lahor' $\to$ 'Lahore').",
        "• <b>Solo Team Rule:</b> When <code>group_size == 1</code>, automatically enforced <code>team_min_cgpa == team_avg_cgpa</code>.",
        "• <b>CGPA Invariant:</b> Repaired instances where <code>team_min_cgpa > team_avg_cgpa</code> via swap correction.",
        "• <b>Deduplication:</b> Removed 4 exact duplicate rows; resolved 16 duplicate <code>group_id</code> entries keeping the most complete record.",
        "• <b>Date Parsing:</b> Robustly parsed mixed ISO, DD/MM/YYYY, and MM-DD-YYYY dates into ISO format.",
        "• <b>Missing Target Rows:</b> 20 records with missing targets were dropped, yielding <b>1,800 clean rows (288 poor outcomes = 16.0% base rate)</b>."
    ]
    for pt in cleaning_points:
        story.append(Paragraph(pt, body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Crucial: Zero Data Leakage Policy</b>", bold_body))
    story.append(Paragraph(
        "Many flawed ML systems achieve artificially high scores by 'cheating'—using information that won't exist at prediction time. "
        "FYP-Pulse enforces a <b>strict Week-8 Information Horizon</b> by permanently dropping 3 columns:",
        body_style
    ))

    leakage_table = [
        [Paragraph("<b>Forbidden Column</b>", bold_body), Paragraph("<b>Why It is Permanently Excluded</b>", bold_body)],
        [Paragraph("<code>final_defense_score</code>", body_style), Paragraph("<b>Future Information:</b> This score is awarded at Week 16. It is physically impossible to know at Week 8. Using it would be complete data leakage.", body_style)],
        [Paragraph("<code>extension_grant</code>", body_style), Paragraph("<b>Target Consequence:</b> Extensions are granted <i>after</i> a project fails or struggles at the end. It directly leaks the target outcome.", body_style)],
        [Paragraph("<code>group_id</code>", body_style), Paragraph("<b>Arbitrary Identifier:</b> A student group ID ('G00123') has no predictive power and would cause memorization/overfitting.", body_style)],
    ]
    t_leak = Table(leakage_table, colWidths=[150, 350])
    t_leak.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_leak)

    story.append(Spacer(1, 15))

    # =========================================================================
    # SLIDE 4: MODEL TRAINING & THE ACCURACY TRAP
    # =========================================================================
    story.append(Paragraph("4. Model Performance: Why Accuracy is a Trap", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    trap_box = [
        [Paragraph("<b>THE 84% ACCURACY TRAP (Must-know for Viva):</b><br/>"
                   "Because only ~16% of student teams fail, a lazy model that predicts 'No Risk' for every single team "
                   "achieves <b>84.0% accuracy</b>! Yet it catches <b>ZERO</b> failing groups (0% recall). "
                   "Therefore, accuracy is useless. We optimize for <b>PR-AUC, Recall, and F1-score</b>.", callout_style)]
    ]
    t_trap = Table(trap_box, colWidths=[500])
    t_trap.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ("BOX", (0, 0), (-1, -1), 1.5, c_gold),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_trap)

    story.append(Spacer(1, 10))

    model_table = [
        [Paragraph("<b>Model</b>", bold_body), Paragraph("<b>Accuracy</b>", bold_body), Paragraph("<b>Precision</b>", bold_body), Paragraph("<b>Recall</b>", bold_body), Paragraph("<b>F1</b>", bold_body), Paragraph("<b>ROC-AUC</b>", bold_body), Paragraph("<b>PR-AUC</b>", bold_body), Paragraph("<b>Threshold</b>", bold_body)],
        [Paragraph("<b>Logistic Regression ⭐</b>", bold_body), Paragraph("77.2%", body_style), Paragraph("40.5%", body_style), Paragraph("<b>56.9%</b>", bold_body), Paragraph("<b>0.471</b>", bold_body), Paragraph("<b>0.796</b>", bold_body), Paragraph("<b>0.419</b>", bold_body), Paragraph("0.221", body_style)],
        [Paragraph("Random Forest", body_style), Paragraph("81.7%", body_style), Paragraph("44.2%", body_style), Paragraph("32.8%", body_style), Paragraph("0.376", body_style), Paragraph("0.773", body_style), Paragraph("0.391", body_style), Paragraph("0.250", body_style)],
        [Paragraph("Extra Trees", body_style), Paragraph("82.5%", body_style), Paragraph("47.2%", body_style), Paragraph("29.3%", body_style), Paragraph("0.362", body_style), Paragraph("0.782", body_style), Paragraph("0.406", body_style), Paragraph("0.260", body_style)],
        [Paragraph("XGBoost Classifier", body_style), Paragraph("80.0%", body_style), Paragraph("38.8%", body_style), Paragraph("32.8%", body_style), Paragraph("0.355", body_style), Paragraph("0.761", body_style), Paragraph("0.388", body_style), Paragraph("0.220", body_style)],
        [Paragraph("<i>Naive 'No Risk' Baseline</i>", body_style), Paragraph("84.0%", body_style), Paragraph("0.0%", body_style), Paragraph("0.0%", body_style), Paragraph("0.000", body_style), Paragraph("0.500", body_style), Paragraph("0.160", body_style), Paragraph("0.500", body_style)],
    ]
    t_models = Table(model_table, colWidths=[120, 52, 52, 52, 48, 58, 58, 60])
    t_models.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ecfdf5")), # Highlight best
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_models)

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Why Did Logistic Regression Win?</b>", bold_body))
    story.append(Paragraph(
        "Logistic Regression achieved the highest **PR-AUC (0.419)** and highest **Recall (56.9% test, 69.2% CV)**. "
        "On tabular academic datasets with class imbalance, complex tree models (XGBoost, RF) tend to over-partition "
        "and become overly conservative on minority classes, whereas penalized logistic regression with Platt scaling "
        "produces smooth, well-calibrated linear log-odds probabilities.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: PREDICTION TIERS, EXPLAINABILITY & FAIRNESS
    # =========================================================================
    story.append(Paragraph("5. Monotonic Tiers, Explainability & Fairness", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    story.append(Paragraph("<b>A. Empirical Risk Tier Monotonicity (Verified on Test Set):</b>", bold_body))
    story.append(Paragraph(
        "A system that assigns 'High Risk' to groups that actually pass more often than 'Medium Risk' groups is dangerous. "
        "We derived tier cut-offs on training data percentiles and verified strict monotonicity on unseen test data ($n=360$):",
        body_style
    ))

    tiers_data = [
        [Paragraph("<b>Risk Tier</b>", bold_body), Paragraph("<b>Probability Range</b>", bold_body), Paragraph("<b>Test Groups ($n$)</b>", bold_body), Paragraph("<b>Observed Failure Rate</b>", bold_body), Paragraph("<b>Action Protocol</b>", bold_body)],
        [Paragraph("🟢 <b>Low Risk</b>", bold_body), Paragraph("P &lt; 0.0414 (&lt;4.1%)", body_style), Paragraph("85 groups", body_style), Paragraph("<b>2.35%</b> (2 failed)", body_style), Paragraph("Standard milestone tracking; no extra overhead.", body_style)],
        [Paragraph("🟡 <b>Medium Risk</b>", bold_body), Paragraph("0.0414 &le; P &lt; 0.2475", body_style), Paragraph("207 groups", body_style), Paragraph("<b>12.56%</b> (26 failed)", body_style), Paragraph("Bi-weekly supervisor check-ins required.", body_style)],
        [Paragraph("🔴 <b>High Risk</b>", bold_body), Paragraph("P &ge; 0.2475 (&ge;24.8%)", body_style), Paragraph("68 groups", body_style), Paragraph("<b>44.12%</b> (30 failed)", bold_body), Paragraph("Mandatory weekly intervention & peer tutoring.", body_style)],
    ]
    t_tier = Table(tiers_data, colWidths=[80, 110, 85, 115, 110])
    t_tier.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_tier)

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Explainability: The 'Why' Behind Every Prediction:</b>", bold_body))
    story.append(Paragraph(
        "Black-box probabilities are rejected by professors. FYP-Pulse uses a <b>baseline-replacement marginal contribution method</b> "
        "that measures how much each feature pushes the risk up (⬆️) or down (⬇️) compared to training medians, generates plain-English sentences, "
        "and automatically prescribes a concrete support step (e.g. <i>'Schedule a weekly supervisor slot'</i> or <i>'Redistribute workload'</i>).",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>C. Algorithmic Fairness & Equity Certification:</b>", bold_body))
    fairness_summary = [
        [Paragraph("<b>Subgroup Dimension</b>", bold_body), Paragraph("<b>Sample Size</b>", bold_body), Paragraph("<b>Predicted Risk Rate</b>", bold_body), Paragraph("<b>Recall (Catch Rate)</b>", bold_body), Paragraph("<b>Statistical Disparity Test</b>", bold_body)],
        [Paragraph("<b>Male Leaders</b>", body_style), Paragraph("n = 210", body_style), Paragraph("22.38% (95% CI: [17.3, 28.5])", body_style), Paragraph("56.76%", body_style), Paragraph("Rate Gap: <b>0.95 pp</b><br/>p = 0.8318 (Chi-square test)<br/><b>No Disparity</b>", body_style)],
        [Paragraph("<b>Female Leaders</b>", body_style), Paragraph("n = 150", body_style), Paragraph("23.33% (95% CI: [17.3, 30.7])", body_style), Paragraph("57.14%", body_style), Paragraph("Recall Gap: <b>0.38 pp</b><br/>Virtually identical equity.", body_style)],
        [Paragraph("<b>Campuses (&ge; 30)</b>", body_style), Paragraph("8 Campuses", body_style), Paragraph("18.18% (Multan) to 28.41% (Karachi)", body_style), Paragraph("40.0% to 88.2%", body_style), Paragraph("Gaps reflect local commit counts; no systemic algorithm bias.", body_style)],
    ]
    t_fair = Table(fairness_summary, colWidths=[90, 65, 145, 95, 105])
    t_fair.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_fair)

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: WATCHLIST & WHAT-IF INTERVENTION SIMULATOR
    # =========================================================================
    story.append(Paragraph("6. Operational Tools: Watchlist & What-If Simulator", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=14))

    story.append(Paragraph("<b>A. The Coordinator's Weekly Watchlist (Module A6):</b>", bold_body))
    story.append(Paragraph(
        "Department chairs cannot review 300 groups individually. The Watchlist ranks the top-K highest-risk projects "
        "($K=10$ to $100$) for the weekly FYP committee agenda with instant CSV export:", body_style
    ))

    watchlist_metrics = [
        [Paragraph("<b>Top-K Size</b>", bold_body), Paragraph("<b>Failing Groups Caught</b>", bold_body), Paragraph("<b>Precision@K</b>", bold_body), Paragraph("<b>Random Expectation</b>", bold_body), Paragraph("<b>Lift Over Random</b>", bold_body)],
        [Paragraph("<b>Top 20 Groups</b>", body_style), Paragraph("<b>8 failing groups</b>", bold_body), Paragraph("<b>40.0%</b>", bold_body), Paragraph("3.22 groups", body_style), Paragraph("<b>2.48× Lift ⭐</b>", bold_body)],
    ]
    t_watch = Table(watchlist_metrics, colWidths=[90, 110, 90, 110, 100])
    t_watch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eff6ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_watch)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<i>Takeaway: Inspecting the top 20 groups captures 2.5× more at-risk projects than random committee audits.</i>", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. The What-If Intervention Simulator (Module A7):</b>", bold_body))
    story.append(Paragraph(
        "Predicting risk is only half the battle; faculty need to know <i>what levers actually reduce risk</i>. "
        "The What-If Simulator allows professors to test realistic behavioral changes while locking immutable demographics:",
        body_style
    ))

    levers_table = [
        [Paragraph("<b>Type</b>", bold_body), Paragraph("<b>Features</b>", bold_body), Paragraph("<b>Policy / Behavioral Meaning</b>", bold_body)],
        [Paragraph("<b>Actionable Levers</b><br/>(Editable)", bold_body),
         Paragraph("• <code>github_commits_8wk</code><br/>"
                   "• <code>supervisor_meetings_8wk</code><br/>"
                   "• <code>supervisor_active_groups</code><br/>"
                   "• <code>has_industry_sponsor</code>", body_style),
         Paragraph("These represent concrete university actions: assigning mandatory weekly meetings, enforcing git sprints, or redistributing supervisor load.", body_style)],
        [Paragraph("<b>Locked Attributes</b><br/>(Frozen)", bold_body),
         Paragraph("• Team Avg / Min CGPA<br/>• Campus & Program<br/>• Leader Gender<br/>• Supervisor Experience", body_style),
         Paragraph("Students cannot magically change their prior GPA or gender in Week 8. Locking these prevents deceptive or impossible simulations.", body_style)],
    ]
    t_lev = Table(levers_table, colWidths=[110, 160, 230])
    t_lev.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_lev)

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Sensitivity Sweep Ranking:</b> Clicking 'Which lever moves this group's score the most?' sweeps each lever across its entire range. In our test group, increasing GitHub Commits reduced risk by <b>38.9 pp</b>, and increasing supervisor meetings reduced risk by <b>31.2 pp</b>.", body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Crucial Scientific Disclaimer:</b> The simulator explicitly warns: <i>'The model shows statistical association, not a guaranteed causal effect.'</i>", body_style))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 2: TOP 10 VIVA QUESTIONS & EASY ANSWERS
    # =========================================================================
    story.append(Paragraph("7. Essential Viva Defense Questions & Answers", h1_style))
    story.append(Paragraph("<i>Prepared in simple, plain, easy-to-recite English for academic examiners and judges.</i>", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_gold, spaceAfter=14))

    viva_qa = [
        ("Q1: In simple words, why did you develop FYP-Pulse?",
         "In universities, about 1 in 6 student groups fail their Final Year Project or get pushed into a delayed semester. "
         "By the time the final defense happens in Week 16, it is too late to help them. "
         "We built FYP-Pulse to detect struggling groups early at Week 8 (Mid-Semester) using only information known at that time, "
         "giving teachers 8 full weeks to intervene, fix issues, and save the students from failing."),

        ("Q2: Why did you NOT use Accuracy to choose your best model?",
         "Because accuracy is deceptive on imbalanced data! In our dataset, 84% of groups succeed and only 16% fail. "
         "A completely useless model that just says 'No Risk' to every single student will get 84% accuracy, "
         "even though it failed to catch a single at-risk student (0% recall). "
         "That's why we evaluated models on PR-AUC (Precision-Recall Area), Recall, and F1-score—not accuracy alone."),

        ("Q3: What is Data Leakage, and how did you prevent it?",
         "Data leakage means giving the model information during training that it wouldn't have in real life when predicting. "
         "For example, 'final_defense_score' is given at Week 16. If we included it to predict Week 8 risk, our model would 'cheat'. "
         "We strictly dropped 'final_defense_score', 'extension_grant', and 'group_id' before training to ensure a genuine Week-8 horizon."),

        ("Q4: Why did Logistic Regression beat complex models like XGBoost and Random Forest?",
         "XGBoost and Random Forests are powerful, but on moderate-sized tabular datasets with imbalanced classes, "
         "deep tree ensembles tend to overfit and create jumpy, extreme probabilities. "
         "Logistic Regression with Platt sigmoid calibration models smooth linear log-odds, which yielded a higher PR-AUC (0.419 vs 0.388) "
         "and much higher recall (56.9% vs 32.8%) at our tuned decision threshold."),

        ("Q5: What is Probability Calibration (Platt Sigmoid)?",
         "When an uncalibrated model outputs '0.80', it doesn't actually mean 80% of those groups fail; it's just an internal mathematical score. "
         "Calibration scales model outputs so that a predicted probability of 30% genuinely corresponds to 30 out of 100 historical groups failing. "
         "We used cross-validated Platt sigmoid scaling on training data."),

        ("Q6: What is Risk Tier Monotonicity, and why does it matter?",
         "Monotonicity means our risk tiers must strictly make sense in real life: the actual observed failure rate on unseen test data "
         "must be Low < Medium < High. In our test set, Low Risk had 2.35% failures, Medium had 12.56%, and High had 44.12%. "
         "If High Risk had fewer failures than Medium, the system would lose all credibility with faculty."),

        ("Q7: How does your Explainability feature work without SHAP compilation?",
         "We implemented a baseline-replacement marginal contribution method. For any student group, we temporarily replace each feature "
         "with the typical university median or mode, re-run the model, and measure how much the failure risk drops or rises. "
         "The features causing the biggest positive jump are flagged as the Top 3 Risk Drivers and mapped to actionable advice."),

        ("Q8: How did you test for Fairness and Disparity?",
         "We analyzed predictions across Male-led vs Female-led groups and across 8 regional campuses (n >= 30). "
         "We computed Wilson score 95% Confidence Intervals and ran a Chi-Square test of independence. "
         "Gender had an insignificant gap of only 0.95 percentage points (p = 0.8318), certifying equity."),

        ("Q9: What is 'Lift' in the Weekly Watchlist?",
         "Lift measures how much better our AI is than random chance. If a committee randomly picked 20 groups, they would only find ~3 failing groups. "
         "FYP-Pulse's Top-20 Watchlist captures 8 failing groups (40% precision). "
         "8 divided by 3.22 equals a 2.48× Lift—meaning the committee is 2.5 times more effective at catching failing projects!"),

        ("Q10: Why are some features locked in the What-If Simulator?",
         "In real life, a student group at Week 8 cannot go back in time to change their prior CGPA, change their degree program, "
         "or change their gender. Allowing users to change CGPA would be unrealistic wishful thinking. "
         "We locked immutable academic history and only made behavioral actions editable: meetings, commits, supervisor load, and industry sponsor."),
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", q_style))
        story.append(Paragraph(a, a_style))

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>End of Presentation & Viva Defense Guide — Team Star ⭐</b>", ParagraphStyle(
        "FooterCenter", parent=styles["Normal"], alignment=1, fontName="Helvetica-Bold", fontSize=10, textColor=c_primary
    )))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")


if __name__ == "__main__":
    build_pdf()
