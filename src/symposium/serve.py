"""서버 실행 엔트리포인트."""
import uvicorn


def main():
    # 127.0.0.1 바인딩: LAN/Tailscale 직접 노출 차단. 공개는 오직 로컬 리버스프록시/터널 경유.
    uvicorn.run("symposium.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
