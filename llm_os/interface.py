from emoji import emojize

class CLIInterface:
    @staticmethod
    def warning_message(msg: str, end='\n'):
        print(emojize(f':warning: {msg}'), end=end, flush=True)
    
    @staticmethod
    def internal_monologue(msg: str, end='\n'):
        print(emojize(f':thought_balloon: {msg}'), end=end, flush=True)

    @staticmethod
    def assistant_message(msg: str, end='\n'):
        print(emojize(f':robot: {msg}'), end=end, flush=True)
    
    @staticmethod
    def memory_message(msg: str, end='\n'):
        print(emojize(f':brain: {msg}'), end=end, flush=True)

    @staticmethod
    def system_message(msg: str, end='\n'):
        print(emojize(f':desktop_computer: {msg}'), end=end, flush=True)

    @staticmethod
    def user_message(msg: str, end='\n'):
        print(emojize(f':person: {msg}'), end=end, flush=True)

    @staticmethod
    def append_to_message(msg: str):
        print(msg, end='', flush=True)