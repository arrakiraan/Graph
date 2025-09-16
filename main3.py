from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

app = FastAPI()

os.makedirs("charts", exist_ok=True)

# Cell 2: Load and prepare dataset
data = pd.read_excel('student_marks_advanced.xlsx')
subject_cols = ['Maths', 'Science', 'English', 'History', 'Computer']

data['TotalMarks'] = data[subject_cols].sum(axis=1)
data['AverageMarks'] = data['TotalMarks'] / len(subject_cols)
data['Result'] = np.where((data[subject_cols] >= 40).all(axis=1), 'Pass', 'Fail')

print(data.head())

app.mount("/charts", StaticFiles(directory="charts"), name="charts")


@app.get("/students")
def get_students():
    """Return dataset as JSON"""
    return JSONResponse(content=data.to_dict(orient="records"))

# Cell 4: Grouped Bar Chart - Average Subject Marks by Gender
@app.get("/chart/avg_marks_by_gender")
def avg_marks_by_gender():

    plt.figure(figsize=(10,6))
    avg_by_gender = data.groupby('Gender')[subject_cols].mean()
    avg_by_gender.T.plot(kind='bar', figsize=(10,6))
    plt.title('Average Subject Marks by Gender')
    plt.ylabel('Average Marks')
    plt.xticks(rotation=0)
    plt.legend(title='Gender')
    plt.tight_layout()
    filepath = "avg_marks_by_gender.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)

# Cell 5: Stacked Bar Chart - Pass/Fail Distribution by Class
@app.get("/chart/Pass/pass_fail_by_class")
def pass_fail_by_class():

    result_counts = data.groupby(['Class','Result']).size().unstack(fill_value=0)
    result_counts.plot(kind='bar', stacked=True, color=['green','red'])
    plt.title('Pass/Fail Distribution by Class')
    plt.ylabel('Number of Students')
    plt.xlabel('Class')
    plt.legend(title='Result')
    plt.tight_layout()
    filepath = "pass_fail_by_class.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)


# Cell 6: Scatter Plot w/Regression Line - Maths vs Science
@app.get("/chart/maths_science_regression")
def maths_science_regression():
    
    plt.figure(figsize=(8,6))
    sns.regplot(x='Maths', y='Science', data=data, scatter_kws={'color':'blue'}, line_kws={'color':'orange'})
    plt.title('Maths vs Science Marks (w/Regression)')
    plt.xlabel('Maths Marks')
    plt.ylabel('Science Marks')
    plt.tight_layout()
    filepath ="maths_science_regression.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)

# Cell 7: Boxplot - AverageMarks by Class
@app.get("/chart/boxplot_avgmarks_by_class")
def boxplot_avgmarks_by_class():

    plt.figure(figsize=(8,6))
    sns.boxplot(x='Class', y='AverageMarks', data=data, palette='Set3')
    plt.title('Average Marks Distribution by Class')
    plt.ylabel('Average Marks')
    plt.tight_layout()
    filepath ="boxplot_avgmarks_by_class.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)

# Cell 8: Histogram + KDE - TotalMarks
@app.get("/chart/totalmarks_hist_kde")
def totalmarks_hist_kde():

    plt.figure(figsize=(8,6))
    sns.histplot(data['TotalMarks'], bins=10, kde=True, color='skyblue')
    plt.title('Distribution of Total Marks (w/KDE)')
    plt.xlabel('Total Marks')
    plt.ylabel('Frequency')
    plt.tight_layout()
    filepath ="totalmarks_hist_kde.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)


# Cell 9: Radar Chart (Spider Plot) - 3 Students
def make_spider_chart(student_names):
    num_vars = len(subject_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7,7))
    for name in student_names:
        student = data[data['Name']==name].iloc[0]
        values = student[subject_cols].tolist()
        values += values[:1]
        plt.polar(angles, values, label=name)
    plt.xticks(angles[:-1], subject_cols)
    plt.title('Student Subject Performance (Radar)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('radar_chart_students.png')
    plt.close()

# Pick Top, Average, Weak performers
top_student = data.loc[data['TotalMarks'].idxmax()]['Name']
weak_student = data.loc[data['TotalMarks'].idxmin()]['Name']
avg_student = data.iloc[(data['TotalMarks'] - data['TotalMarks'].mean()).abs().idxmin()]['Name']
make_spider_chart([top_student, avg_student, weak_student])

# Cell 10: Heatmap of Subject Correlations
@app.get("/chart/heatmap_subject_correlation")
def heatmap_subject_correlation():

    plt.figure(figsize=(8,6))
    corr_matrix = data[subject_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu')
    plt.title('Correlation Matrix: Subjects')
    plt.tight_layout()
    filepath ="heatmap_subject_correlation.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)



# Cell 11: Pie Chart - Gender Distribution
@app.get("/chart/pie_gender_distribution")
def pie_gender_distribution():

    plt.figure(figsize=(5,5))
    gender_counts = data['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue','pink'])
    plt.title('Student Gender Distribution')
    plt.tight_layout()
    filepath = "pie_gender_distribution.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)

# Pie (Donut) Chart - Pass vs Fail Distribution
@app.get("/chart/pie_result_distribution")
def pie_result_distribution():
    
    plt.figure(figsize=(5,5))
    result_counts = data['Result'].value_counts()
    plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', colors=['lime','red'], wedgeprops=dict(width=0.5))
    plt.title('Student Result Distribution')
    plt.tight_layout()
    filepath = "pie_result_distribution.png"
    plt.savefig(filepath)
    plt.close()
    return FileResponse(filepath)

