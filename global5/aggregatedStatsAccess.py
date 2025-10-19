import re
from collections import defaultdict

# Define patterns to search for statistics
#patterns = {
#    "TCP Hits": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.m_demand_hits\s+(\d+)",
#    "TCP Misses": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.m_demand_misses\s+(\d+)",
#    "TCC Hits": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.m_demand_hits\s+(\d+)",
#    "TCC Misses": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.m_demand_misses\s+(\d+)",
#    "L1 GPU TLB Hits": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.tlb\.hits\s+(\d+)",
#    "L1 GPU TLB Misses": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.tlb\.misses\s+(\d+)",
#    "L2 GPU TLB Hits": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.tlb\.hits\s+(\d+)",
#    "L2 GPU TLB Misses": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.tlb\.misses\s+(\d+)",
#    "m_demand_accesses": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.m_demand_accesses\s+(\d+)"
#}

patterns = {
    # Data cache stats (what you already have)
    "TCP Data Hits": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.m_demand_hits\s+(\d+)",
    "TCP Data Misses": r"system\.ruby\.tcp_cntrl\d+\.L1cache\.m_demand_misses\s+(\d+)",
    "TCC Data Hits": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.m_demand_hits\s+(\d+)",
    "TCC Data Misses": r"system\.ruby\.tcc_cntrl\d+\.L2cache\.m_demand_misses\s+(\d+)",

    # TLB stats (THESE ARE THE IMPORTANT ONES)
    "L1 TLB Accesses": r"system\.l1_tlb\d+\.globalNumTLBAccesses\s+(\d+)",
    "L1 TLB Hits": r"system\.l1_tlb\d+\.globalNumTLBHits\s+(\d+)",
    "L1 TLB Misses": r"system\.l1_tlb\d+\.globalNumTLBMisses\s+(\d+)",
    "L2 TLB Accesses": r"system\.l2_tlb\.globalNumTLBAccesses\s+(\d+)",
    "L2 TLB Hits": r"system\.l2_tlb\.globalNumTLBHits\s+(\d+)",
    "L2 TLB Misses": r"system\.l2_tlb\.globalNumTLBMisses\s+(\d+)",
    "Page Table Cycles": r"system\.l2_tlb\.pageTableCycles\s+(\d+)",
    "L2 Access Cycles": r"system\.l2_tlb\.accessCycles\s+(\d+)",
}

def process_stats(filename):
    groups = []
    current_group = []
    stats = defaultdict(list)
    
    with open(filename, "r") as file:
        for line in file:
            if "Begin Simulation Statistics" in line:
                current_group = []
            elif "End Simulation Statistics" in line:
                groups.append(current_group)
            else:
                current_group.append(line.strip())
    
    # Remove first and last group
    if len(groups) > 2:
        groups = groups[1:-1]
    
    # Process statistics in groups of 10
    for i in range(0, len(groups), 10):
        aggregated = defaultdict(int)
        for j in range(10):
            if i + j < len(groups):
                for line in groups[i + j]:
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            aggregated[key] += int(match.group(1))
        stats[i // 10] = aggregated
    
    return stats

stats = process_stats("stats.txt")

# Print aggregated stats
for batch, data in stats.items():
    print(f"Batch {batch}: ")
    for key, value in data.items():
        print(f"  {key}: {value}")

