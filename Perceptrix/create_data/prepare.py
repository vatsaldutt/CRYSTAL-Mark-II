import pyperclip


while True:
    user = input("Enter the new user: ")
    json_format = """
        "instruction": "{}",
        "input": "{}: {}",
        "output": "CRYSTAL: {}"
    """
    
    with open("instructions.txt", 'r') as ins_file:
        instructions = ins_file.read()

    with open("input.txt", 'r') as input_file:
        input_ = input_file.read()

    with open("output.txt", 'r') as output_file:
        output = output_file.read()

    json_format = json_format.format(instructions.replace('\n', '\\n').replace('"', '\\"'), user,  input_.replace("\n", "\\n").replace('"', '\\"'), output.replace("\n", '\\n').replace('"', '\\"'))

    
    print("{"+json_format+"}")
    pyperclip.copy("{"+json_format+"}")