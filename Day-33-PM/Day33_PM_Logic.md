# Day 33 PM – Session Logic

## Dataset Choice
Synthetic 8-feature dataset with ~50/50 class balance and non-linear decision boundary.
This lets all 8 algorithms have a fair chance to show their strengths.

## Part A Design
Each algorithm gets:
- When to use (1 sentence)
- 3 key parameters
- 1 pro, 1 con
- 5-line code snippet (copy-paste ready)

Fair comparison uses 5-fold CV with identical splits for all models.
Pipelines handle scaling where needed.

## Part B Design
20newsgroups is the standard text benchmark.
LinearSVC + TF-IDF is the classic pipeline.
Comparing with Logistic Regression shows SVM's strength on text data.

## Part C Design
- Q1: 100 features / 50 samples forces curse-of-dimensionality thinking.
- Q2: model_selection_report() is production-ready with t-test for statistical significance.
- Q3: SVM overfitting is diagnosed with specific parameter fixes.

## Part D Design
AI-generated decision tree is verified against real Week 6 experience.
Edge cases are added based on common failure modes seen in assignments.
