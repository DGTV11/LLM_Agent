from emoji import emojize

class CLIInterface:
    @staticmethod
    def warning_message(msg: str):
        print(emojize(f':warning: {msg}'))
    
    @staticmethod
    def internal_monologue(msg: str):
        print(emojize(f':thought_balloon: {msg}'))

    @staticmethod
    def assistant_message(msg: str):
        print(emojize(f':robot: {msg}'))
    
    @staticmethod
    def memory_message(msg: str):
        print(emojize(f':brain: {msg}'))

    @staticmethod
    def system_message(msg: str):
        print(emojize(f':desktop_computer: {msg}'))

    @staticmethod
    def user_message(msg: str):
        print(emojize(f':person: {msg}'))
