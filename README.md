# Financial News Triage OpenEnv

This project implements a deterministic OpenEnv environment for financial headline triage.

The agent reads a headline and predicts:

- relevance
- sector
- urgency
- action

Rewards are calculated with exact literal matching across those four fields.

## What is in this repo

- `env/`: core environment logic (`reset`, `step`, `state`), loader, grader, models
- `data/`: `easy.json`, `medium.json`, `hard.json` datasets
- `app.py`: FastAPI server with `/reset`, `/step`, `/state`
- `inference.py`: deterministic baseline runner
- `openenv.yaml`, `pyproject.toml`, `uv.lock`: OpenEnv packaging metadata
- `server/app.py`: packaging bridge entrypoint

## Quick start (local)

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn app:app --reload
```

4. Smoke test the endpoints:

```bash
curl -X POST http://127.0.0.1:8000/reset
curl -X POST http://127.0.0.1:8000/step \
	-H "Content-Type: application/json" \
	-d '{"relevance":"high","sector":"macro","urgency":"high","action":"buy"}'
curl http://127.0.0.1:8000/state
```

## Baseline policy

The baseline in `inference.py` is intentionally simple and deterministic:

- fixed action each step: `high / macro / high / buy`
- max 8 steps
- strict log format for validator compatibility

Run it with:

```bash
python inference.py
```

## Validation

From the project root:

```bash
openenv validate
```

## Notes

- No random policy behavior is used in the environment loop.
- Dataset loading and grading are deterministic.
- API is intentionally minimal for validator compatibility.
