import os
import re

def verify_set_calculation(vaddr, page_shift, set_mask, reported_set):
    """Verify if the set calculation matches what's reported"""
    calculated_set = (vaddr >> page_shift) & set_mask
    return calculated_set == reported_set, calculated_set

def parse_hex(hex_str):
    """Parse hex string to integer, handling '0x' prefix"""
    return int(hex_str, 16) if hex_str.startswith('0x') else int(hex_str, 16)

def analyze_tlb_sets(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Only analyze lines related to L2 TLB
    l2_lines = [line for line in content.split('\n') 
                if 'l2_tlb' in line and 
                ('Translation Hit' in line or 'Translation Miss' in line)]
    
    results = []
    pattern = r"vaddr=(0x[0-9a-fA-F]+).*\(([0x0-9a-fA-F]+)\s*>>\s*(\d+)\)\s*&\s*([0x0-9a-fA-F]+)\s*=\s*(\d+)"
    
    for line in l2_lines:
        match = re.search(pattern, line)
        if match:
            vaddr = parse_hex(match.group(1))
            shift_val = int(match.group(3))
            mask = parse_hex(match.group(4))
            reported_set = int(match.group(5))
            
            is_correct, calculated_set = verify_set_calculation(vaddr, shift_val, mask, reported_set)
            
            results.append({
                'line': line,
                'vaddr': hex(vaddr),
                'page_shift': shift_val,
                'set_mask': hex(mask),
                'reported_set': reported_set,
                'calculated_set': calculated_set,
                'is_correct': is_correct,
                'type': 'Hit' if 'Translation Hit' in line else 'Miss'
            })
    
    return results

def print_summary(results):
    """Print summary of the analysis"""
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    hits = sum(1 for r in results if r['type'] == 'Hit')
    misses = sum(1 for r in results if r['type'] == 'Miss')
    
    print(f"\nAnalysis Summary:")
    print(f"Total calculations analyzed: {total}")
    print(f"Correct calculations: {correct}")
    print(f"Incorrect calculations: {total - correct}")
    print(f"Total Hits: {hits}")
    print(f"Total Misses: {misses}")
    
    # Print set distribution
    set_dist = {}
    for r in results:
        set_dist[r['reported_set']] = set_dist.get(r['reported_set'], 0) + 1
    
    print("\nSet Distribution:")
    for set_num, count in sorted(set_dist.items()):
        print(f"Set {set_num}: {count} accesses")

def main():
    # Process all split files
    for filename in sorted(os.listdir('.')):
        if filename.startswith('global_section_') and filename.endswith('.txt'):
            print(f"\nAnalyzing {filename}:")
            results = analyze_tlb_sets(filename)
            
            # Print detailed results
            for r in results:
                status = "✓" if r['is_correct'] else "✗"
                print(f"\n{status} {r['type']} - Virtual Address: {r['vaddr']}")
                print(f"  Set Calculation: ({r['vaddr']} >> {r['page_shift']}) & {r['set_mask']}")
                print(f"  Reported Set: {r['reported_set']}")
                print(f"  Calculated Set: {r['calculated_set']}")
            
            print_summary(results)

if __name__ == "__main__":
    main()
