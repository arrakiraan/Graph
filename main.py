from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Literal
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

# JSON
class StudentMark(BaseModel):
    name: str
    score: float

@app.post("/plot-marks/")
async def plot_marks(
    data: List[StudentMark],
    chart_type: Literal["bar", "line", "pie"] = Query("bar")
):
    names = [d.name for d in data] #list comprehensio
    scores = [d.score for d in data]

    plt.figure(figsize=(6, 4))

    if chart_type == "bar":
        plt.bar(names, scores, color="skyblue")

        plt.title("Bar Chart - Student Marks")
        plt.xlabel("Students")
        plt.ylabel("Scores")

    elif chart_type == "line":
        plt.plot(names, scores, marker="o", linestyle="-", color="green")
        
        plt.title("Line Chart - Student Marks")
        plt.xlabel("Students")
        plt.ylabel("Scores")

    elif chart_type == "pie":
        plt.pie(scores, labels=names, autopct="%1.1f%%", startangle=140)
        plt.title("Pie Chart - Student Marks")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")