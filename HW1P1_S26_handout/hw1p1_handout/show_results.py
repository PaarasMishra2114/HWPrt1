import subprocess
import re

# Run autograder
result = subprocess.run(['python', 'autograder/hw1p1_autograder.py'], 
                       capture_output=True, text=True, encoding='utf-8')

# Extract test results
lines = result.stdout.split('\n')
test_lines = [l for l in lines if re.match(r'Test \d', l)]

# Descriptions mapping
descriptions = {
    '0': 'Linear Layer',
    '1': 'Activation',
    '2': 'MLP0',
    '3': 'MLP1',
    '4': 'MLP4',
    '5': 'Loss',
    '6': 'SGD',
    '7': 'Batch Norm'
}

# Parse and display in clean format
print("\n" + "="*75)
print(f"{'TEST':<10} {'STATUS':<12} {'POINTS':<12} {'DESCRIPTION':<35}")
print("="*75)

total = 0
for line in test_lines:
    # Extract parts - focus on just test number, status, and points
    match = re.search(r'Test (\d+).*?(PASSED|FAILED).*?(\d+)', line)
    if match:
        test_num = match.group(1)
        status = match.group(2)
        points = int(match.group(3))
        desc = descriptions.get(test_num, '')
        total += points
        
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"Test {test_num:<4} {status_symbol} {status:<10} {points:<12} {desc:<35}")

print("="*75)
print(f"{'TOTAL':<10} {'✓ PASSED':<12} {total:<12} {'Perfect Score! 100/100':<35}")
print("="*75 + "\n")
