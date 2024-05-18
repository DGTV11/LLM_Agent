import configparser

def get_config():
    config = configparser.ConfigParser()
    config.read(path.join(path.dirname(__file__), 'config.ini'))
    
    server_url = config.get('Server', 'server_url')
    model_name = config.get('Models', 'llm_name')
    embedding_name = config.get('Models', 'embedding_name') 
    ctx_window = config.getint('ContextWindow', 'size')

    return {
        'server_url': server_url,
        'llm_name': llm_name,
        'embedding_name': embedding_name,
        'ctx_window': ctx_window
    }

CONFIG = get_config()
