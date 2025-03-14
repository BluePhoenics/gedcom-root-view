import math
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from ged4py import GedcomReader
import numpy as np

###############################################################################
# 0) Settings
###############################################################################

max_gen = 12 # Maximum generations to display in the tree

gedcom_file = "test-family.ged"  # Input genealogy file

main_family_id = "@F1@"  # Entry point family (parents of the root person)

output_figsize = (270, 150)

###############################################################################
# 1) Parse GEDCOM File
###############################################################################
def parse_gedcom(file_path):
    """Extract individuals and families from GEDCOM file.
    Returns:
    individuals: {id: {name, sex}} 
    families: {id: {husb, wife, chil[], marr_date}}
    """
    individuals = {}
    families = {}

    with GedcomReader(file_path) as parser:
        # Process individual records
        for record in parser.records0("INDI"):
            indi_id = record.xref_id
            name = record.name.format() if record.name else "N.N."
            sex = record.sex if record.sex else "U"
            individuals[indi_id] = {"name": name, "sex": sex}

        # Process family records
        for fam_record in parser.records0("FAM"):
            fam_id = fam_record.xref_id
            husb_tag = fam_record.sub_tag("HUSB")
            husband_id = husb_tag.xref_id if husb_tag else None

            wife_tag = fam_record.sub_tag("WIFE")
            wife_id = wife_tag.xref_id if wife_tag else None

            chil_tags = fam_record.sub_tags("CHIL")
            child_ids = [c.xref_id for c in chil_tags]

            # Extract marriage date if available
            marr_tag = fam_record.sub_tag("MARR")
            if marr_tag:
                date_tag = marr_tag.sub_tag("DATE")
                if date_tag and date_tag.value:
                    marr_date_str = str(date_tag.value)
                else:
                    marr_date_str = None
            else:
                marr_date_str = None

            families[fam_id] = {
                "husb": husband_id,
                "wife": wife_id,
                "chil": child_ids,
                "marr_date": marr_date_str
            }

    return individuals, families

###############################################################################
# 2) Build Family Tree Structure
###############################################################################
def find_family_of_person(person_id, families):
    """Find which family a person belongs to as a child"""
    for fam_id, fam_data in families.items():
        if person_id in fam_data["chil"]:
            return fam_id
    return None

def build_ancestry_tree(main_family_id, families, max_gen=5):
    """Recursively build tree structure starting from target family.
    Returns nested dictionary with family nodes and their parents."""
    visited = set() # Prevent infinite loops

    def build_recursive(fam_id, current_gen):
        if fam_id in visited:
            return None
        visited.add(fam_id)

        if current_gen >= max_gen:
            return None

        # Get parent families for current family's spouses
        father_id = families[fam_id]["husb"]
        mother_id = families[fam_id]["wife"]
        father_fam = find_family_of_person(father_id, families) if father_id else None
        mother_fam = find_family_of_person(mother_id, families) if mother_id else None

        # Build node with recursive parent lookup
        node = {"family_id": fam_id, "parents": []}
        next_gen = current_gen + 1

        if father_fam:
            pnode = build_recursive(father_fam, next_gen)
            if pnode: node["parents"].append(pnode)

        if mother_fam:
            pnode = build_recursive(mother_fam, next_gen)
            if pnode: node["parents"].append(pnode)

        return node

    return build_recursive(main_family_id, 0)

###############################################################################
# 3) Calculate Node Positions
###############################################################################
def layout_ancestry_bfs(tree, families, max_radius_step=2.0):
    """Calculate positions using radial layout:
    - Generations as concentric circles
    - Nodes spaced by angles within generation
    Returns positions for individuals and family nodes."""
    positions_ind = {}
    positions_fam = {}
    generation_map = defaultdict(list)
    
    # BFS to group nodes by generation
    queue = deque([(tree, 0)])
    while queue:
        node, gen = queue.popleft()
        generation_map[gen].append(node)
        for parent_node in node["parents"]:
            queue.append((parent_node, gen + 1))

    # Polar coordinates parameters
    angle_start = -math.pi / 2
    angle_end = math.pi / 2
    angle_range = angle_end - angle_start

    def get_text_angle(theta):
        angle_deg = math.degrees(theta) + 90
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        return angle_deg

    def coords_and_angle(r, theta):
        x_ = r * math.sin(theta)
        y_ = -r * math.cos(theta)
        a_ = get_text_angle(theta)
        return x_, y_, a_

    all_gens = sorted(generation_map.keys())
    placed_individuals = set()

    # Calculate positions for each generation
    for gen in all_gens:
        fam_nodes_this_gen = generation_map[gen]
        if not fam_nodes_this_gen:
            continue

        count_father = sum(1 for fam in fam_nodes_this_gen if families[fam["family_id"]]["husb"])
        count_mother = sum(1 for fam in fam_nodes_this_gen if families[fam["family_id"]]["wife"])
        count_emptynodes = sum(1 for fam in fam_nodes_this_gen if (not families[fam["family_id"]]["wife"] and not families[fam["family_id"]]["husb"]))
        count_nodes = sum(1 for fam in fam_nodes_this_gen)
        
        sector_size = angle_range / max(1, (count_father + count_mother + count_emptynodes))
        sector_size_node = angle_range / (2 * count_nodes)
        
        for i, fam_node in enumerate(fam_nodes_this_gen):
            fam_id = fam_node["family_id"]
            father_id = families[fam_id]["husb"]
            mother_id = families[fam_id]["wife"]
            child_ids = families[fam_id]["chil"]

            if gen == max_gen-1:
                angle_mid = angle_start + (2*i + 0.5) * sector_size_node
            else:
                angle_mid = angle_start + 2*(i + 0.5) * sector_size
            alpha_start = angle_mid - sector_size
            alpha_end = angle_mid + sector_size

            if child_ids:
                step_kids = (alpha_end - alpha_start) / (len(child_ids) + 1)
                alpha = alpha_start + step_kids
                for cid in child_ids:
                    if cid not in placed_individuals:
                        if gen == 0:
                            xC, yC, aC = coords_and_angle(-0.5 * max_radius_step, alpha)
                        else:
                            xC, yC, aC = coords_and_angle(gen * max_radius_step, alpha)
                        positions_ind[cid] = (xC, yC, aC)
                        alpha += step_kids
                        placed_individuals.add(cid)

            angle_mid_node = angle_start + 2*(i + 0.5) * sector_size_node
            fm_radius = 2* (0.35 + 1.045* gen - 0.0018* gen*gen)
            fx, fy, fangle = coords_and_angle(fm_radius, angle_mid_node)
            positions_fam[fam_id] = (fx, fy, fangle)

            
            if gen == max_gen-1:
                nxt_angle_mid = angle_start + 2*(i + 0.5) * sector_size_node
                alpha_f = nxt_angle_mid - 0.25*sector_size_node
                alpha_m = nxt_angle_mid + 0.25*sector_size_node
            else:
                nxt_angle_mid = angle_start + 2*(i + 0.5) * sector_size_node
                alpha_f = nxt_angle_mid - 0.25*sector_size_node
                alpha_m = nxt_angle_mid + 0.25*sector_size_node
                

            if father_id and father_id not in placed_individuals:
                xF, yF, aF = coords_and_angle((gen + 1) * max_radius_step, alpha_f)
                positions_ind[father_id] = (xF, yF, aF)
                placed_individuals.add(father_id)

            if mother_id and mother_id not in placed_individuals:
                xM, yM, aM = coords_and_angle((gen + 1) * max_radius_step, alpha_m)
                positions_ind[mother_id] = (xM, yM, aM)
                placed_individuals.add(mother_id)

    return positions_ind, positions_fam

###############################################################################
# 4) Visualization Helpers
###############################################################################
def extract_year(date_str):
    if not date_str:
        return " "
    tokens = date_str.strip().split()
    for t in tokens:
        if len(t) >= 4 and t[:4].isdigit():
            return t[:4]
    return " "

def plot_bezier_curve(ax, x0, y0, x1, y1, x2, y2, x3, y3,
                      color='black', lw=1, n=50):
    """Draw cubic Bezier curve between points using control points"""
    tvals = np.linspace(0, 1, n)

    X = (1 - tvals)**3 * x0 \
        + 3 * (1 - tvals)**2 * tvals * x1 \
        + 3 * (1 - tvals) * tvals**2 * x2 \
        + tvals**3 * x3

    Y = (1 - tvals)**3 * y0 \
        + 3 * (1 - tvals)**2 * tvals * y1 \
        + 3 * (1 - tvals) * tvals**2 * y2 \
        + tvals**3 * y3

    ax.plot(X, Y, color=color, lw=lw)

def draw_smooth_connection(ax, x1, y1, angle1, x2, y2, angle2,
                           color='black', lw=1, dist=0.3, n=50):

    a1_rad = math.radians(angle1)
    a2_rad = math.radians(angle2)

    cx1 = x1 + dist * math.cos(a1_rad)
    cy1 = y1 + dist * math.sin(a1_rad)
    cx2 = x2 + dist * math.cos(a2_rad)
    cy2 = y2 + dist * math.sin(a2_rad)

    plot_bezier_curve(
        ax,
        x1, y1,  # Start
        cx1, cy1,# Point 1
        cx2, cy2,# Point 2
        x2, y2,  # End
        color=color, lw=lw, n=n
    )

###############################################################################
# 5) Drawing Functions
###############################################################################
def draw_family_tree(positions_ind, positions_fam, individuals, families):
    fig, ax = plt.subplots(figsize=output_figsize)
    """Main visualization function using matplotlib:
    - Draws connections between family members
    - Renders individual/family nodes with annotations"""

    line_width = 2
    dist_bend = 1  # instead of 0.3 => curved, longer Bézier curves
    dist_bend_par = 0.5

    # --- Connections ---
    for fam_id, (fx, fy, fangle) in positions_fam.items():
        father_id = families[fam_id]["husb"]
        mother_id = families[fam_id]["wife"]
        child_ids = families[fam_id]["chil"]

        # Father -> Family
        if father_id in positions_ind:
            fangleaF = fangle
            xF, yF, aF = positions_ind[father_id]
            if fangleaF > 0:
                fangleaF += 180
            if aF < 0:
                aF += 180
            draw_smooth_connection(ax, xF, yF, aF, fx, fy, fangleaF,
                                   color='black', lw=line_width, dist=dist_bend_par)

        # Mother -> Family
        if mother_id in positions_ind:
            fangleaM = fangle
            xM, yM, aM = positions_ind[mother_id]
            if fangleaM > 0:
                fangleaM += 180
            if aM < 0:
                aM += 180
            draw_smooth_connection(ax, xM, yM, aM, fx, fy, fangleaM,
                                   color='black', lw=line_width, dist=dist_bend_par)

        # Child -> Family
        for cid in child_ids:
            if cid in positions_ind:
                fangleaC = fangle
                xC, yC, aC = positions_ind[cid]
                if fangleaC < 0:
                    fangleaC += 180
                if aC > 0:
                    aC += 180
                draw_smooth_connection(ax, xC, yC, aC, fx, fy, fangleaC,
                                       color='black', lw=line_width, dist=dist_bend)

    # --- Family-Box ---
    for fam_id, (fx, fy, fangle) in positions_fam.items():
        full_date = families[fam_id].get("marr_date", None)
        year_str = extract_year(full_date)

        ax.text(
            fx, fy, year_str,
            ha="center",
            va="center",
            fontsize=7,  # größerer Font
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="lightgreen",
                ec="black",
                lw=1,
                alpha=0.6
            )
        )

    # --- Person as Box ---
    for person_id, (x, y, angle_deg) in positions_ind.items():
        sex = individuals[person_id]["sex"]
        if sex == "M":
            color = "blue"
        elif sex == "F":
            color = "red"
        else:
            color = "gray"

        name = individuals[person_id]["name"]
        ax.text(
            x, y, name,
            fontsize=12,  # größerer Font
            ha='center',
            va='center',
            rotation=angle_deg,
            rotation_mode='anchor',
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc=color,
                ec="black",
                lw=1,
                alpha=0.5
            )
        )

    ax.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    plt.show()

###############################################################################
# 6) MAIN
###############################################################################
if __name__ == "__main__":
    # Data pipeline:
    # 1. Parse input file
    # 2. Build tree structure
    # 3. Calculate positions
    # 4. Render visualization

    individuals, families = parse_gedcom(gedcom_file)
    ancestry_tree = build_ancestry_tree(main_family_id, families, max_gen)
    positions_ind, positions_fam = layout_ancestry_bfs(ancestry_tree, families)
    draw_family_tree(positions_ind, positions_fam, individuals, families)
