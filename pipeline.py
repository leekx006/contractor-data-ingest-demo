import re
import io
from datetime import date
from pdfminer.high_level import extract_text
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import anthropic

# ── Semantic Matching ─────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticMatcher:
    _instance = None

    @classmethod
    def get(cls):
        """Singleton — load model once and reuse."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, threshold=0.72):
        self.threshold = threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def best_match(self, text, candidates):
        """
        Returns the best matching candidate string if similarity
        exceeds threshold, otherwise None.
        """
        if not text or not candidates:
            return None
        text_emb  = self.model.encode([text])
        cand_embs = self.model.encode(candidates)
        scores    = cosine_similarity(text_emb, cand_embs)[0]
        best_idx  = int(np.argmax(scores))
        if scores[best_idx] >= self.threshold:
            return candidates[best_idx]
        return None


# ── Color objects (for TableStyle) ───────────────────────
DARK_BLUE  = colors.HexColor("#1F4E79")
LIGHT_BLUE = colors.HexColor("#D6E4F0")
GREEN_BG   = colors.HexColor("#E2EFDA")
GREEN_FG   = colors.HexColor("#375623")
AMBER_BG   = colors.HexColor("#FFF2CC")
AMBER_FG   = colors.HexColor("#7F6000")
RED_BG     = colors.HexColor("#FFE0E0")
RED_FG     = colors.HexColor("#C00000")
LIGHT_GRAY = colors.HexColor("#F5F5F5")

# ── Hex strings (for Paragraph markup) ───────────────────
GREEN_HEX = "#375623"
AMBER_HEX = "#7F6000"
RED_HEX   = "#C00000"

STATUS_COLORS = {
    "APPROVED":     (GREEN_FG,  GREEN_BG,  GREEN_HEX),
    "CONDITIONAL":  (AMBER_FG,  AMBER_BG,  AMBER_HEX),
    "DISQUALIFIED": (RED_FG,    RED_BG,    RED_HEX),
}
RAG_COLORS = {
    "GREEN": (GREEN_FG, GREEN_BG, GREEN_HEX),
    "AMBER": (AMBER_FG, AMBER_BG, AMBER_HEX),
    "RED":   (RED_FG,   RED_BG,   RED_HEX),
}

# ── Company label variations ──────────────────────────────
COMPANY_LABELS = [
    "Legal Name", "Legal Entity Name", "Legal Company Name",
    "Company Name", "Business Name", "Contractor Name",
    "DBA", "Trade Name", "Also Known As", "Doing Business As",
    "Federal Tax ID", "Tax ID Number", "IRS EIN", "FEIN",
    "Federal Employer ID", "EIN",
    "Year Established", "Established", "In Business Since",
    "Founded", "Year Founded",
    "Headquarters", "Company Location", "Main Office",
    "Where We Operate", "Business Address", "Primary Location",
    "Office Location", "Corporate Address",
    "Field Employees", "Number of Employees", "Field Workforce",
    "How Many Workers", "Total Employees", "Craft Employees",
    "Field Workers", "Field Personnel",
    "Office Staff", "Administrative Staff", "Office Personnel",
    "Admin Staff",
    "Union / Non-Union", "Labor Relations", "Union Status",
    "Union or Non-Union",
    "Primary Services", "Work Performed", "What We Do",
    "Services Provided", "Scope of Work", "Trade",
    "Primary Trade", "Services",
    "Annual Revenue", "Yearly Revenue", "Annual Sales", "Revenue",
    "Annual Manhours", "Hours Per Year", "Annual Man Hours",
    "Manhours", "Annual Hours Worked",
    "ISNetworld ID", "ISNetworld", "ISN ID", "ISN Number",
    "Avetta ID", "Avetta", "Avetta Number",
]

# ── Section header variations ─────────────────────────────
EMR_HEADERS = [
    "EMR Documentation", "Experience Modification Rate",
    "EMR History", "Modification Rate",
    "Insurance and Safety Rating", "Workers Compensation Rate",
    "EMR Information", "Experience Mod", "EMR",
]

OSHA_STAT_HEADERS = [
    "OSHA Incident Statistics", "Safety Performance Summary",
    "Safety Incident Summary", "Safety Record",
    "Our Safety Record", "Incident Statistics",
    "Safety Statistics", "OSHA Statistics",
    "Safety Performance", "Injury Statistics",
    "Incident Rates", "OSHA Recordables",
]

OSHA_CITATION_HEADERS = [
    "OSHA Citation History", "Citation History",
    "OSHA Citations", "Regulatory History",
    "Citation Summary", "OSHA Violations",
    "Employee Training", "Training Matrix",
    "Training Records", "Workforce Training",
]


# ── Extraction ────────────────────────────────────────────
def extract_table_columns(lines, known_labels):
    result = {}
    label_set = set(known_labels)
    matcher = SemanticMatcher.get()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Layer 1: exact string match
        matched_label = line if line in label_set else None
        # Layer 2: semantic match fallback
        # Exclude section headers like "2. Primary Contacts"
        if (matched_label is None and len(line) > 3 and len(line) < 60
                and not re.match(r"^\d+\.", line)):
            matched_label = matcher.best_match(line, known_labels)
        if matched_label is not None:
            # Collect consecutive matched labels
            label_block = [matched_label]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Check next line for label match
                if next_line in label_set:
                    next_match = next_line
                elif (len(next_line) > 3 and len(next_line) < 60
                        and not re.match(r"^\d+\.", next_line)):
                    next_match = matcher.best_match(next_line, known_labels)
                else:
                    next_match = None
                if next_match is not None:
                    label_block.append(next_match)
                    j += 1
                else:
                    break
            # Skip section headers between labels and values
            while j < len(lines) and re.match(r"^\d+\.", lines[j]):
                j += 1
            # Skip section headers between labels and values
            while j < len(lines) and re.match(r"^\d+\.", lines[j]):
                j += 1
            # Collect values, merging lowercase continuation lines
            value_block = []
            k = j
            while k < len(lines) and len(value_block) < len(label_block):
                line_val = lines[k]
                if (value_block and line_val
                        and line_val[0].islower()):
                    value_block[-1] = value_block[-1] + " " + line_val
                else:
                    value_block.append(line_val)
                k += 1
            for idx, label in enumerate(label_block):
                if idx < len(value_block):
                    result[label] = value_block[idx]
            i = j + len(label_block)
        else:
            i += 1
    return result


def normalize_company_info(raw):
    """Map alternative label names to standard field names."""
    mapping = {
        "company_name": [
            "Legal Name", "Legal Entity Name", "Legal Company Name",
            "Company Name", "Business Name", "Contractor Name",
        ],
        "dba": [
            "DBA", "Trade Name", "Also Known As", "Doing Business As",
        ],
        "headquarters": [
            "Headquarters", "Company Location", "Main Office",
            "Where We Operate", "Business Address", "Primary Location",
            "Office Location", "Corporate Address",
        ],
        "field_employees": [
            "Field Employees", "Number of Employees", "Field Workforce",
            "How Many Workers", "Total Employees", "Craft Employees",
            "Field Workers", "Field Personnel",
        ],
        "annual_revenue": [
            "Annual Revenue", "Yearly Revenue", "Annual Sales", "Revenue",
        ],
        "annual_manhours": [
            "Annual Manhours", "Hours Per Year", "Annual Man Hours",
            "Manhours", "Annual Hours Worked",
        ],
    }
    result = {}
    for standard_key, alternatives in mapping.items():
        for alt in alternatives:
            if alt in raw:
                result[standard_key] = raw[alt]
                break
        if standard_key not in result:
            result[standard_key] = None
    return result


def extract_contractor_fields(pdf_file):
    """
    Accepts either a file path (string) or a file-like object
    (from Streamlit uploader).
    """
    try:
        if hasattr(pdf_file, 'read'):
            pdf_file = io.BytesIO(pdf_file.read())
        text = extract_text(pdf_file)
    except Exception as e:
        return {"error": str(e)}

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    raw_company_info = extract_table_columns(lines, COMPANY_LABELS)
    company_info = normalize_company_info(raw_company_info)

    def extract_emr():
        emr_start, emr_end = None, None
        for i, line in enumerate(lines):
            if any(x in line for x in EMR_HEADERS):
                emr_start = i
            if emr_start and i > emr_start and any(x in line for x in OSHA_STAT_HEADERS):
                emr_end = i
                break
        if emr_start is None:
            return []
        section = lines[emr_start: emr_end if emr_end else emr_start + 40]
        years    = [l for l in section if re.match(r"^20\d{2}$", l)]
        values   = [l for l in section if re.match(r"^\d+\.\d+$", l)]
        statuses = [l for l in section if
                    l.startswith("Below threshold") or
                    l.startswith("Above threshold") or
                    l.upper().startswith("SIGNIFICANTLY")]
        return [
            {"year": y, "emr": float(v), "status": s}
            for y, v, s in zip(years, values, statuses)
        ]

    def extract_osha_stats():
        OSHA_COL_HEADERS = {
            "Year", "Hours", "Worked", "Recordables",
            "TRIR", "DART", "Lost Time", "Cases", "Fatalities"
        }
        osha_start, osha_end = None, None
        for i, line in enumerate(lines):
            if any(x in line for x in OSHA_STAT_HEADERS):
                osha_start = i + 1
            if osha_start and i > osha_start and any(x in line for x in OSHA_CITATION_HEADERS):
                osha_end = i
                break
        if osha_start is None:
            return []
        section = lines[osha_start: osha_end if osha_end else osha_start + 50]
        data_lines = []
        for line in section:
            if line in OSHA_COL_HEADERS:
                continue
            if re.match(r"^20\d{2}$", line):
                data_lines.append(line)
            elif re.match(r"^\d{1,3}(,\d{3})*$", line):
                data_lines.append(line.replace(",", ""))
            elif re.match(r"^\d+\.\d+$", line):
                data_lines.append(line)
            else:
                if data_lines:
                    break
        years = [l for l in data_lines if re.match(r"^20\d{2}$", l)]
        n_rows = len(years)
        if n_rows == 0 or len(data_lines) != 7 * n_rows:
            return []
        cols = [data_lines[i * n_rows:(i + 1) * n_rows] for i in range(7)]
        years_col, hours_col, rec_col, trir_col, dart_col, lt_col, fat_col = cols
        stats = []
        for i in range(n_rows):
            try:
                stats.append({
                    "year":            years_col[i],
                    "hours_worked":    hours_col[i],
                    "recordables":     int(rec_col[i]),
                    "trir":            float(trir_col[i]),
                    "dart":            float(dart_col[i]),
                    "lost_time_cases": int(lt_col[i]),
                    "fatalities":      int(fat_col[i])
                })
            except (ValueError, IndexError):
                continue
        return stats

    def detect_flags(osha_stats):
        flags = []
        full_text = text.upper()
        checks = [
            ("Willful citation",
             r"WILLFUL\s+CITATION|WILLFUL\s+VIOLATION(?!S ARE ABSENT|S\.?\s+NO)"),
            ("Open OSHA citation",
             r"OPEN\b(?!.*CLOSED)"),
            ("EMR above 1.3",
             r"SIGNIFICANTLY ABOVE THRESHOLD"),
            ("Missing training",
             r"NO TRAINING ON FILE"),
            ("Missing insurance",
             r"NOT PROVIDED"),
        ]
        for label, pattern in checks:
            if re.search(pattern, full_text):
                flags.append(label)
        if any(s["fatalities"] > 0 for s in osha_stats):
            flags.append("Fatality")
        return flags

    emr_history  = extract_emr()
    osha_stats   = extract_osha_stats()
    red_flags    = detect_flags(osha_stats)

    result_match = re.search(
        r"OVERALL RESULT:\s*([^\n]+)", text, re.IGNORECASE)
    overall_result = (result_match.group(1).strip()
                      if result_match else "Not found")

    return {
        "company_name":    company_info.get("company_name") or "Unknown",
        "dba":             company_info.get("dba"),
        "headquarters":    company_info.get("headquarters"),
        "field_employees": company_info.get("field_employees"),
        "annual_revenue":  company_info.get("annual_revenue"),
        "annual_manhours": company_info.get("annual_manhours"),
        "emr_history":     emr_history,
        "osha_stats":      osha_stats,
        "overall_result":  overall_result,
        "red_flags":       red_flags,
        "red_flag_count":  len(red_flags)
    }


# ── Scoring ───────────────────────────────────────────────
def score_contractor(data):
    scores = {}
    rag    = {}
    flags  = data["red_flags"]

    if data["emr_history"]:
        emr = data["emr_history"][0]["emr"]
        if emr < 0.85:
            scores["emr"], rag["emr"] = 15, "GREEN"
        elif emr < 1.0:
            scores["emr"], rag["emr"] = 12, "GREEN"
        elif emr < 1.3:
            scores["emr"], rag["emr"] = 5,  "AMBER"
        else:
            scores["emr"], rag["emr"] = 0,  "RED"
    else:
        scores["emr"], rag["emr"] = 0, "RED"

    if data["osha_stats"]:
        trir = data["osha_stats"][0]["trir"]
        if trir < 1.0:
            scores["trir"], rag["trir"] = 20, "GREEN"
        elif trir < 2.0:
            scores["trir"], rag["trir"] = 16, "GREEN"
        elif trir < 4.0:
            scores["trir"], rag["trir"] = 8,  "AMBER"
        else:
            scores["trir"], rag["trir"] = 0,  "RED"
    else:
        scores["trir"], rag["trir"] = 0, "RED"

    scores["fatality"],  rag["fatality"]  = (
        (0, "RED") if "Fatality" in flags else (10, "GREEN"))
    scores["citations"], rag["citations"] = (
        (0, "RED")  if "Willful citation" in flags else
        (5, "AMBER") if "Open OSHA citation" in flags else
        (15, "GREEN"))
    scores["insurance"], rag["insurance"] = (
        (0, "RED") if "Missing insurance" in flags else (10, "GREEN"))
    scores["training"],  rag["training"]  = (
        (0, "RED") if "Missing training" in flags else (10, "GREEN"))

    total   = sum(scores.values())
    auto_dq = "Fatality" in flags or "Willful citation" in flags

    if auto_dq:
        final_status = "DISQUALIFIED"
    elif total >= 65:
        final_status = "APPROVED"
    elif total >= 40:
        final_status = "CONDITIONAL"
    else:
        final_status = "DISQUALIFIED"

    return {
        "scores": scores, "rag": rag,
        "total": total, "max_score": 80,
        "auto_dq": auto_dq, "final_status": final_status
    }


# ── Claude Explanation ────────────────────────────────────
def generate_explanation(company_name, scoring, data, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        flags    = data["red_flags"]
        emr_val  = data["emr_history"][0]["emr"] if data["emr_history"] else "N/A"
        trir_val = data["osha_stats"][0]["trir"]  if data["osha_stats"]  else "N/A"
        fat_val  = data["osha_stats"][0]["fatalities"] if data["osha_stats"] else "N/A"

        prompt = f"""
You are a contractor qualification analyst for an industrial safety platform.
Based on the following data, write a concise 3-4 sentence explanation of why
this contractor received their qualification result. Address key strengths or
concerns directly. Write in plain English suitable for a compliance reviewer.
Do not use bullet points.

Contractor: {company_name}
Final Status: {scoring['final_status']}
Total Score: {scoring['total']} out of {scoring['max_score']}
Auto-Disqualified: {scoring['auto_dq']}
EMR Score: {scoring['scores']['emr']}/15 ({scoring['rag']['emr']})
TRIR Score: {scoring['scores']['trir']}/20 ({scoring['rag']['trir']})
Fatality History: {scoring['scores']['fatality']}/10 ({scoring['rag']['fatality']})
Citation History: {scoring['scores']['citations']}/15 ({scoring['rag']['citations']})
Insurance: {scoring['scores']['insurance']}/10 ({scoring['rag']['insurance']})
Training: {scoring['scores']['training']}/10 ({scoring['rag']['training']})
Most Recent EMR: {emr_val}
Most Recent TRIR: {trir_val}
Fatalities (most recent year): {fat_val}
Active Risk Flags: {', '.join(flags) if flags else 'None'}
"""
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


# ── PDF Generation ────────────────────────────────────────
def make_styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("HdrTitle",
        fontName="Helvetica-Bold", fontSize=16,
        textColor=colors.white, alignment=TA_CENTER))
    s.add(ParagraphStyle("HdrSub",
        fontName="Helvetica", fontSize=9,
        textColor=colors.white, alignment=TA_CENTER))
    s.add(ParagraphStyle("CoName",
        fontName="Helvetica-Bold", fontSize=14,
        textColor=DARK_BLUE, spaceAfter=2))
    s.add(ParagraphStyle("CoInfo",
        fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#444444"), spaceAfter=2))
    s.add(ParagraphStyle("SecHdr",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=DARK_BLUE, spaceBefore=10, spaceAfter=4))
    s.add(ParagraphStyle("Cell",
        fontName="Helvetica", fontSize=9))
    s.add(ParagraphStyle("Footer",
        fontName="Helvetica", fontSize=7,
        textColor=colors.gray, alignment=TA_CENTER))
    return s


def sp(h=6):
    return Spacer(1, h)


def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=LIGHT_BLUE, spaceAfter=4)


def th_para(text):
    return Paragraph(text, ParagraphStyle(
        "TH", fontName="Helvetica-Bold", fontSize=9,
        textColor=colors.white, alignment=TA_CENTER))


def rag_label(rag):
    _, bg, fg_hex = RAG_COLORS[rag]
    labels = {"GREEN": "PASS", "AMBER": "REVIEW", "RED": "FAIL"}
    return Paragraph(
        f'<font color="{fg_hex}"><b>{labels[rag]}</b></font>',
        ParagraphStyle("rl", fontName="Helvetica-Bold",
                       fontSize=9, backColor=bg))


def score_bar(score, max_pts, rag):
    _, _, fg_hex = RAG_COLORS[rag]
    pct = int((score / max_pts) * 100) if max_pts else 0
    return Paragraph(
        f'<font color="{fg_hex}"><b>{score}/{max_pts} ({pct}%)</b></font>',
        ParagraphStyle("sb", fontName="Helvetica-Bold", fontSize=9))


def base_table_style():
    return TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK_BLUE),
        ("GRID",          (0,0), (-1,-1), 0.4,
                          colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ])


def generate_scorecard_pdf(company_name, data, scoring, explanation=None):
    """
    Generates a scorecard PDF and returns it as bytes
    so Streamlit can offer it as a download.
    """
    s = make_styles()
    buffer = io.BytesIO()

    final_status = scoring["final_status"]
    status_fg, status_bg, status_hex = STATUS_COLORS[final_status]

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.65*inch, rightMargin=0.65*inch,
        topMargin=0.65*inch,  bottomMargin=0.65*inch)
    story = []

    # Header
    hdr_table = Table([
        [Paragraph("CONTRACTOR QUALIFICATION SCORECARD", s["HdrTitle"])],
        [Paragraph(
            f"CanQualify Intelligence Platform  |  "
            f"Generated {date.today().strftime('%B %d, %Y')}",
            s["HdrSub"])]
    ], colWidths=[7.2*inch])
    hdr_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK_BLUE),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story += [hdr_table, sp(8)]

    # Company info
    story.append(Paragraph(company_name, s["CoName"]))
    info_parts = [
        f"<b>DBA:</b> {data.get('dba') or 'N/A'}",
        f"<b>HQ:</b> {data.get('headquarters') or 'N/A'}",
        f"<b>Employees:</b> {data.get('field_employees') or 'N/A'}",
        f"<b>Revenue:</b> {data.get('annual_revenue') or 'N/A'}",
        f"<b>Manhours:</b> {data.get('annual_manhours') or 'N/A'}",
    ]
    story.append(Paragraph("   |   ".join(info_parts), s["CoInfo"]))
    story += [sp(6), hr()]

    # Verdict banner
    auto_note = "  (AUTO-DISQUALIFIED)" if scoring["auto_dq"] else ""
    verdict_table = Table([[
        Paragraph(
            f'<font color="{status_hex}"><b>{final_status}'
            f'{auto_note}</b></font>',
            ParagraphStyle("vl", fontName="Helvetica-Bold",
                           fontSize=14, alignment=TA_LEFT)),
        Paragraph(
            f'<font color="{status_hex}"><b>Total Score: '
            f'{scoring["total"]} / {scoring["max_score"]}</b></font>',
            ParagraphStyle("vr", fontName="Helvetica-Bold",
                           fontSize=14, alignment=TA_RIGHT)),
    ]], colWidths=[3.6*inch, 3.6*inch])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), status_bg),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
    ]))
    story += [verdict_table, sp(10)]

    # Scores table
    story.append(Paragraph("Qualification Scores", s["SecHdr"]))
    SCORE_ROWS = [
        ("EMR",                 "emr",       15),
        ("Safety Stats / TRIR", "trir",       20),
        ("Fatality History",    "fatality",   10),
        ("Citation History",    "citations",  15),
        ("Insurance",           "insurance",  10),
        ("Training",            "training",   10),
    ]
    score_data = [[th_para("Category"),
                   th_para("Score"),
                   th_para("Status")]]
    for label, key, max_pts in SCORE_ROWS:
        rag = scoring["rag"][key]
        score_data.append([
            Paragraph(label, s["Cell"]),
            score_bar(scoring["scores"][key], max_pts, rag),
            rag_label(rag),
        ])
    score_table = Table(score_data,
                        colWidths=[3.2*inch, 2.4*inch, 1.6*inch])
    score_table.setStyle(base_table_style())
    story += [score_table, sp(10)]

    # EMR History
    story.append(Paragraph("EMR History", s["SecHdr"]))
    emr_data = [[th_para("Policy Year"),
                 th_para("EMR"),
                 th_para("Status")]]
    for row in data["emr_history"]:
        emr_val = float(row["emr"])
        fg_hex = (GREEN_HEX if emr_val < 1.0
                  else AMBER_HEX if emr_val < 1.3
                  else RED_HEX)
        emr_data.append([
            Paragraph(str(row["year"]), s["Cell"]),
            Paragraph(
                f'<font color="{fg_hex}"><b>{emr_val:.2f}</b></font>',
                s["Cell"]),
            Paragraph(str(row["status"]), s["Cell"]),
        ])
    if len(emr_data) > 1:
        emr_table = Table(emr_data,
                          colWidths=[1.6*inch, 1.6*inch, 4.0*inch])
        emr_table.setStyle(base_table_style())
        story += [emr_table, sp(10)]
    else:
        story.append(Paragraph(
            "EMR data could not be extracted from this document.",
            ParagraphStyle("warn", fontName="Helvetica",
                           fontSize=9, textColor=AMBER_FG)))
        story.append(sp(10))

    # OSHA Statistics
    story.append(Paragraph("OSHA Incident Statistics", s["SecHdr"]))
    osha_data = [[th_para("Year"), th_para("Hours Worked"),
                  th_para("Recordables"), th_para("TRIR"),
                  th_para("DART"), th_para("Fatalities")]]
    for row in data["osha_stats"]:
        trir = float(row["trir"])
        fat  = int(row["fatalities"])
        trir_hex = (GREEN_HEX if trir < 2.0
                    else AMBER_HEX if trir < 4.0
                    else RED_HEX)
        fat_hex = RED_HEX if fat > 0 else GREEN_HEX
        osha_data.append([
            Paragraph(str(row["year"]),         s["Cell"]),
            Paragraph(str(row["hours_worked"]), s["Cell"]),
            Paragraph(str(row["recordables"]),  s["Cell"]),
            Paragraph(
                f'<font color="{trir_hex}"><b>{trir}</b></font>',
                s["Cell"]),
            Paragraph(str(row["dart"]),         s["Cell"]),
            Paragraph(
                f'<font color="{fat_hex}"><b>{fat}</b></font>',
                s["Cell"]),
        ])
    if len(osha_data) > 1:
        osha_table = Table(osha_data,
                           colWidths=[0.9*inch, 1.1*inch, 1.0*inch,
                                      0.9*inch, 0.9*inch, 0.9*inch])
        osha_ts = base_table_style()
        osha_ts.add("ALIGN", (1,1), (-1,-1), "CENTER")
        osha_table.setStyle(osha_ts)
        story += [osha_table, sp(10)]
    else:
        story.append(Paragraph(
            "OSHA statistics could not be extracted from this document.",
            ParagraphStyle("warn", fontName="Helvetica",
                           fontSize=9, textColor=AMBER_FG)))
        story.append(sp(10))

    # Risk Flags
    story.append(Paragraph("Risk Flags", s["SecHdr"]))
    if data["red_flags"]:
        for flag in data["red_flags"]:
            story.append(Paragraph(
                f'<font color="{RED_HEX}"><b>WARNING: {flag}</b></font>',
                ParagraphStyle("flag", fontName="Helvetica-Bold",
                               fontSize=9, spaceAfter=3)))
    else:
        story.append(Paragraph(
            f'<font color="{GREEN_HEX}">No critical issues identified.</font>',
            ParagraphStyle("ok", fontName="Helvetica-Bold", fontSize=9)))

    # AI Explanation
    if explanation:
        story += [sp(6), hr()]
        story.append(Paragraph("AI Analysis", s["SecHdr"]))
        story.append(Paragraph(explanation,
            ParagraphStyle("exp", fontName="Helvetica",
                           fontSize=9, leading=14,
                           textColor=colors.HexColor("#333333"),
                           spaceAfter=6)))

    # Footer
    story += [sp(12), hr()]
    story.append(Paragraph(
        "CONFIDENTIAL — Generated by CanQualify Intelligence Platform "
        "| Powered by UT (Ultra Tendency) | For internal review only.",
        s["Footer"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()