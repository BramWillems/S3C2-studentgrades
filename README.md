# Predicting Student Mastery with Bayesian Knowledge Tracing and Deep Learning

**Live demo:** https://bramstudentgradeapp.streamlit.app/

## The problem

Student performance is usually measured at a single point in time — an exam.
But a single exam can produce false positives: a student might get lucky or
unlucky, or fail a class despite having actually mastered the material. This
project estimates a mastery score per skill instead of relying on one test
result, giving a broader picture of what a student actually knows. As a
side effect, the model also reveals how difficult each skill is to master.

## Result

Out of 187 skills fitted, 35 produced reliable results, with around 70%
prediction accuracy on those skills. Even among the reliable skills, mastery
difficulty varies a lot — for example, skill 169 was extremely hard, with
only about 5% of students reaching mastery on it.

The live app above uses a simplified version of the dataset, but produces
results consistent with the full analysis in the notebook.

## Data

The dataset is [EdNet](https://github.com/riiid/ednet), which contains roughly a million
students' interactions (~10GB) with a standardized English test in South
Korea, in a multiple-choice format. Because it's a standardized test, the
relative differences in performance between students should generalize
reasonably well beyond this one population.

## Method

- Engineered features to capture: whether an answer was correct, how
  difficult a given skill is, and what proportion of students had mastered
  a skill
- Trained a Bayesian Knowledge Tracing model, which updates its estimate of
  a student's mastery after every interaction
- Compared this against a deep learning model and evaluated both approaches
  against each other

## What I learned

Results don't have to match expectations to be useful — most skills (152 of
187) didn't produce reliable estimates, but the 35 that did still gave
meaningful, actionable insight. A partial result can still be a viable one.
