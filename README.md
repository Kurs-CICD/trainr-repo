# TrainR

TrainR is an AI fitness coach that recommends exercises, meals and daily training actions using machine learning, deep learning and reinforcement learning.

It started as my university dissertation, built in a single Colab notebook. I'm now rebuilding it properly as a full-stack web app, in public, one week at a time.

## Status

🚧 **Rebuild in progress: Week 1 of 22** (Git and GitHub fundamentals). Progress is tracked on the project board and in the weekly issues.

## What it will do

- **Exercise recommendations** from a classifier trained on about 2,900 exercises
- **Meal suggestions** that fit a user's calorie and macro targets
- **A daily training action** (train hard, rest, adjust calories) from a reinforcement learning agent that learns from energy, soreness and progress check-ins

## Planned stack

| Layer | Tools |
|---|---|
| ML / DL / RL | scikit-learn, Keras, Gymnasium, MLflow |
| Backend | FastAPI, Pydantic, SQLAlchemy 2.0, Alembic |
| Database | MySQL 8 |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Testing | pytest, Vitest, Playwright |
| DevOps | Docker, Docker Compose, GitHub Actions |

## Repository layout

```
ml/        machine learning package (data, models, RL environment)
backend/   FastAPI service
frontend/  React + TypeScript app
docs/      design notes, decisions and learning log
```

## How I work

- `main` is protected: every change goes through a branch and a pull request
- Each week of the plan has its own [issue](https://github.com/Kurs-CICD/trainr-repo/issues?q=label%3Aweek), linked to the pull requests that close it

## Licence

MIT. See [LICENSE](LICENSE).
