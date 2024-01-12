import subprocess

def main():
    for text_number in range(32, 100):
        subprocess.run(["bun", "run", "pull-content.ts", f"./texts/text_{text_number}.txt"])
        print(f"Added content text_{text_number}.txt")

if __name__ == '__main__':
    main()
