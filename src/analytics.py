# Took Logic from Bo_Nix_Target_Accuracy
# Need to rebuild to be applicable to any quarterback, however might need to start from beginning
# because we are moving entirely to nflreadpy and this project was stupidly built with nfL_data_py

def classify_throw(description):
    description = description.upper()
    if "SHORT" in description or "SWING" in description or "SCREEN" in description or "QUICK" in description or "SLANT" in description:
        return "Short"
    elif "MEDIUM" in description or "MID" in description or "CROSS" in description or "DIG" in description or "OUT ROUTE" in description:
        return "Medium"
    elif "DEEP" in description or "BOMB" in description or "FADE" in description or "GO ROUTE" in description or "POST" in description or "CORNER" in description:
        return "Deep"
    else:
        return "Unknown"

#Pass location classification (more robust)
def pass_location(description):
    description = description.upper()
    if "LEFT" in description:
        return "Left"
    elif "MIDDLE" in description or "CENTER" in description:
        return "Middle"
    elif "RIGHT" in description:
        return "Right"
    else:
        return "Unknown"

def completion(description):
    description = description.upper()
    if "INCOMPLETE" in description or "DROPS" in description or "INTERCEPTED" in description:
        return 0
    elif "SHORT RIGHT TO" in description or "SHORT LEFT TO" in description or "SHORT MIDDLE TO" in description or "DEEP RIGHT TO" in description or "DEEP LEFT TO" in description or "DEEP MIDDLE TO" in description:
        return 1
    else:
        return None