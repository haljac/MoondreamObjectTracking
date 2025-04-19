import zenoh
import random
import threading
import time
import signal

random.seed()
running = True

def create_config():
    config = zenoh.Config()
    
    # Connection endpoints
    config.insert_json5(
        "connect/endpoints", 
        '["tcp/localhost:7447"]'  # Must be valid JSON5 string
    )
    
    # Multicast scouting configuration
    config.insert_json5("scouting/multicast/enabled", "true")
    config.insert_json5("scouting/multicast/address", '"224.0.0.224:7447"')
    
    # TCP-specific settings
    config.insert_json5("transport/link/tcp/so_rcvbuf", "65535")
    config.insert_json5("transport/link/tcp/so_sndbuf", "65535")
    
    return config

def read_temp():
    return random.randint(15, 30)

def sub(session):
    print("Subscriber starting...")
    subscriber = session.declare_subscriber('myhome/kitchen/temp', sub_handler)
    while running:
        time.sleep(0.1)
    subscriber.undeclare()

def sub_handler(sample: zenoh.Sample):
    print(f"[SUB] Temperature update: {sample.payload.to_string()}")

def pub(session):
    print("Publisher starting...")
    key = 'myhome/kitchen/temp'
    pub = session.declare_publisher(key)
    while running:
        t = read_temp()
        pub.put(f"{t}")
        time.sleep(1)
    pub.undeclare()

def signal_handler(sig, frame):
    global running
    running = False
    print("\nShutting down...")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    with zenoh.open(create_config()) as session:
        threads = [
            threading.Thread(target=sub, args=(session,)),
            threading.Thread(target=pub, args=(session,))
        ]
        
        for t in threads:
            t.start()
            
        while running:
            time.sleep(1)
            
        for t in threads:
            t.join()
