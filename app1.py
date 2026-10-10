from dashboard import app, start_background_services


if __name__ == "__main__":
    start_background_services()
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
