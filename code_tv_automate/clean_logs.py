import os

input_folder = 'C:\\Users\\u255769\\tv_mobile_data\\tv_data\\data\\P1-1489\\P1-1489012_data\\P1-1489012_data\\txts'

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)

        with open(file_path, 'rb') as file:
            content = file.read()

        cleaned_content = content.replace(b'\x00', b'')
        original_lines = content.splitlines()
        cleaned_lines = cleaned_content.splitlines()

        print(f"\nChanges in file: {filename}")
        for i, (orig_line, clean_line) in enumerate(zip(original_lines, cleaned_lines)):
            if orig_line != clean_line:
                print(f"Line {i + 1} changed:")
                print(f"Original: {orig_line}")
                print(f"Cleaned:  {clean_line}")

        with open(file_path, 'wb') as file:
            file.write(cleaned_content)

print("All files processed and overwritten.")
