"""Convert docs/report.html to docs/report.docx"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from html2docx import html2docx

html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "report.html")
out_path  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "Report_NgoHoangHuy_EEEEIU21019.docx")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

buf = html2docx(html, title="GRU Autoencoder-Based Semantic Compression and Anomaly Detection on IoT Gateway")

with open(out_path, "wb") as f:
    f.write(buf.getvalue())

print(f"Saved: {out_path}")
print(f"Size : {os.path.getsize(out_path) / 1024:.1f} KB")
