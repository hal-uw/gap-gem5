import re
import sys
from pathlib import Path

def extract_set_calculation(line):
    """Extract the set calculation from a line of code"""
    # Common patterns for set calculations
    patterns = [
        r'set\s*=\s*\((.*?)\)\s*&\s*setMask',
        r'set\s*=\s*(.*?)\s*&\s*setMask'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1).strip()
    return None

def analyze_file(filename):
    """Analyze a single file for set calculations"""
    print(f"\n{'='*50}")
    print(f"Analyzing file: {filename}")
    print(f"{'='*50}\n")
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return

    # Find all set-related variable definitions
    var_definitions = {
        'setMask': None,
        'numSets': None,
        'assoc': None,
        'size': None,
        'PageShift': None
    }

    # Extract variable definitions
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if 'setMask' in line and '=' in line:
            var_definitions['setMask'] = line
        elif 'numSets' in line and '=' in line:
            var_definitions['numSets'] = line
        elif 'assoc' in line and '=' in line and not 'assert' in line:
            var_definitions['assoc'] = line
        elif 'size' in line and ('=' in line or 'size(' in line):
            var_definitions['size'] = line
        elif 'PageShift' in line:
            var_definitions['PageShift'] = line

    # Find all set calculations
    set_calculations = []
    for i, line in enumerate(lines, 1):
        if ('set =' in line or 'set=' in line) and 'setMask' in line:
            calc = extract_set_calculation(line)
            if calc:
                set_calculations.append({
                    'line_number': i,
                    'full_line': line.strip(),
                    'calculation': calc
                })

    # Print findings
    print("1. Variable Definitions Found:")
    for var, def_line in var_definitions.items():
        if def_line:
            print(f"   {var}: {def_line}")
    print()

    print("2. Set Calculations Found:")
    for calc in set_calculations:
        print(f"   Line {calc['line_number']}: {calc['full_line']}")
        print(f"   Extracted calculation: {calc['calculation']}")
        print()

    # Analyze correctness
    print("3. Analysis:")
    
    # Check if all necessary components are present
    required_components = ['setMask', 'numSets', 'assoc']
    missing_components = [comp for comp in required_components 
                         if var_definitions[comp] is None]
    
    if missing_components:
        print("   WARNING: Missing required components:", missing_components)
    
    # Verify calculation pattern
    expected_pattern = ">> PageShift"
    calculations_correct = True
    
    for calc in set_calculations:
        if ">> PageShift" not in calc['calculation']:
            calculations_correct = False
            print(f"   WARNING: Line {calc['line_number']} may have incorrect page shift")
            
    # Check for proper masking
    if var_definitions['setMask'] and 'numSets - 1' not in var_definitions['setMask']:
        print("   WARNING: setMask might not be properly defined as numSets - 1")

    # Print verification result
    print("\n4. Verification Result:")
    if calculations_correct and not missing_components:
        print("   ✓ All set calculations appear to be correct")
        print("   ✓ All required components are present")
        print("   ✓ Proper page shifting is used")
        if var_definitions['setMask'] and 'numSets - 1' in var_definitions['setMask']:
            print("   ✓ Proper set masking is used")
    else:
        print("   ✗ Some issues were found (see warnings above)")

    # Show example calculation if we have all components
    if not missing_components:
        print("\n5. Example Calculation:")
        print("   For virtual address 0x1234000:")
        print("   set = (0x1234000 >> 12) & (numSets - 1)")
        print("   where:")
        print("   - PageShift = 12 (typical for 4KB pages)")
        print("   - numSets = size/assoc")
        print("   This ensures the set index is properly aligned and masked")

if __name__ == "__main__":
    files = [
        "global_section_3.txt",
        "global_section_4.txt",
        "global_section_5.txt",
        "global_section_6.txt"
    ]
    
    for filename in files:
        if Path(filename).exists():
            analyze_file(filename)
        else:
            print(f"\nFile not found: {filename}")
