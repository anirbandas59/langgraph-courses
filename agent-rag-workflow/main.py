from dotenv import load_dotenv

from graph.graph import app

load_dotenv()


def main():
    print(f"{'*' * 10} ADVANCED RAG {'*' * 10}")
    print(app.invoke(input={"question": "What is agent memory?"}))


if __name__ == "__main__":
    main()
