import io
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from mlProject.components.monitoring import PredictionLogger
from mlProject import logger

def get_analytics_summary() -> dict:
    logger_db = PredictionLogger()
    df = logger_db.get_logged_predictions()
    if df.empty:
        return {
            "prediction_count": 0,
            "mean_prediction": 0,
            "min_prediction": 0,
            "max_prediction": 0,
            "daily_trends": []
        }
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    stats = {
        "prediction_count": len(df),
        "mean_prediction": float(df['prediction'].mean()),
        "min_prediction": float(df['prediction'].min()),
        "max_prediction": float(df['prediction'].max())
    }
    
    daily = df.groupby(df['timestamp'].dt.date)['prediction'].agg(['count', 'mean']).reset_index()
    daily['date'] = daily['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    trends = []
    for idx, row in daily.iterrows():
        trends.append({
            "date": row['date'],
            "count": int(row['count']),
            "mean": float(row['mean'])
        })
    stats["daily_trends"] = trends
    return stats

def generate_pdf_report() -> io.BytesIO:
    stats = get_analytics_summary()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6B1D2F'),
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=10
    )
    
    story = []
    
    story.append(Paragraph("Red Wine IQ - Enterprise Analytics Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC) | Model Status: Active", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = (
        f"This enterprise prediction analytics report provides insights into model usage and prediction trends. "
        f"A total of <b>{stats['prediction_count']}</b> predictions have been logged in the monitoring system. "
        f"The average predicted wine quality is <b>{stats['mean_prediction']:.2f}</b> (min: {stats['min_prediction']:.1f}, max: {stats['max_prediction']:.1f})."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Prediction Statistics Summary", heading_style))
    data = [
        ["Metric", "Value"],
        ["Total Volume", str(stats['prediction_count'])],
        ["Average Quality Prediction", f"{stats['mean_prediction']:.4f}"],
        ["Minimum Score", f"{stats['min_prediction']:.2f}"],
        ["Maximum Score", f"{stats['max_prediction']:.2f}"]
    ]
    t = Table(data, colWidths=[200, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#6B1D2F')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F9FAFB')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Data Validation & Drift Summary", heading_style))
    drift_text = (
        "Kolmogorov-Smirnov statistical tests run regularly against the baseline training distribution. "
        "Baseline reference points: artifacts/reference_data.csv. All drift checks are automatically updated "
        "on the live monitoring console."
    )
    story.append(Paragraph(drift_text, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer
