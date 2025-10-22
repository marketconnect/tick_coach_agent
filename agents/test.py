if __name__ == "__main__":
    from agents.message_handler import process_user_message

    user_task = "Создай мне тренировку."

    print(f"🧑 Human: {user_task}\n")
    print("🤖 Assistant:")

    full_response = ""
    # process_user_message is a generator, so we iterate over its chunks
    for chunk in process_user_message(user_task):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n\n--- End of Stream ---")