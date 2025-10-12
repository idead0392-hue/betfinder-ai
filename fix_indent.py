#!/usr/bin/env python3
"""
Fix the indentation issue in prizepicks_scrape.py
"""

def fix_indentation():
    with open('prizepicks_scrape.py', 'r') as f:
        lines = f.readlines()
    
    # Fix line with df.to_csv that has wrong indentation
    for i, line in enumerate(lines):
        if 'df.to_csv(output_csv, index=False)' in line:
            # Replace with correct 12-space indentation
            lines[i] = '            df.to_csv(output_csv, index=False)\n'
            print(f"Fixed line {i+1}: {repr(line.rstrip())} -> {repr(lines[i].rstrip())}")
            break
    
    with open('prizepicks_scrape.py', 'w') as f:
        f.writelines(lines)
    
    print("✅ Fixed indentation issue")

if __name__ == "__main__":
    fix_indentation()