with open('validation_output.txt', 'rb') as f:
    content = f.read()
    # Decode as UTF-16LE and print as ASCII, ignoring errors
    text = content.decode('utf-16le', errors='ignore')
    print(text)
