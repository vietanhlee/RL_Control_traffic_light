import uvicorn

def main() -> int:
    print("Khởi chạy Backend (FastAPI + Simulation)...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
