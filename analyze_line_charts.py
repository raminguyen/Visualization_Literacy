import pandas as pd

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# === 1. Load Line Chart Rows from CSV ===
def load_line_rows(file_path='line_rows.csv'):
    df = pd.read_csv(file_path)
    return df

# === 2. Show number of unique questions ===
def count_unique_questions(df):
    questions = df['question'].unique()
    print(f"The number of questions in the line dataset is {len(questions)}")
    return questions

# === 3. Print list of all unique questions ===
def list_questions(df):
    questions = df['question'].unique().tolist()
    print("List of questions in the line dataset:")
    for question in questions:
        print(question)

# === 4. Count number of rows for each question ===
def question_distribution(df, save_to_csv=True):
    question_counts = df.groupby('question').size().reset_index(name='row_count')
    print(question_counts)
    if save_to_csv:
        question_counts.to_csv('question_counts.csv', index=False)
    return question_counts

# === 5. Extract rows for a target question ===
def get_matching_question_rows(df, target_question):
    matching = df[df['question'] == target_question]
    return matching

# === 6. Check for duplicate rows ===
def check_duplicates(df):
    duplicates = df.duplicated(keep=False)
    print("Duplicate rows:")
    print(df[duplicates])
    return duplicates

# === 7. Analyze color usage ===
def color_usage_summary(df):
    color_cols = [col for col in df.columns if col.startswith("color_")]
    color_usage = df[color_cols].sum()
    used_colors = color_usage[color_usage > 0]
    print("Colors used in the rows:")
    print(used_colors)
    return used_colors

# === 8. List used colors per row ===
def list_used_colors(df):
    color_cols = [col for col in df.columns if col.startswith("color_")]
    df['used_colors'] = df[color_cols].apply(lambda row: [col for col in color_cols if row[col]], axis=1)
    print(df[['filename', 'used_colors']])
    return df

# === 9. Filter rows where color_red is used ===
def filter_color_red_rows(df):
    red_rows = df[df['color_red'] == True]
    print(red_rows)
    return red_rows

