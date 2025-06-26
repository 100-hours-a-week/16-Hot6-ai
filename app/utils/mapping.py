def convert_coords_to_position(x: int, y: int) -> str:
    h_zones = ['left', 'center', 'right']
    v_zones = ['top', 'middle', 'bottom']

    h_index = min(x * 3 // 1024, 2)  # 0~2
    v_index = min(y * 3 // 1024, 2)  # 0~2

    return f"{v_zones[v_index]}-{h_zones[h_index]}"

def format_location_info_natural(obj_pos_dict: dict[str, list[tuple[int, int]]]) -> str:
    lines = []
    for obj, positions in obj_pos_dict.items():
        for idx, (x, y) in enumerate(positions, start=1):
            pos_desc = convert_coords_to_position(x, y)
            lines.append(f"{obj} {idx}번은 책상에서 {pos_desc} 위치에 있습니다.")
    return "\n".join(lines)