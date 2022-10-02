def _deposit_pic(field, hci, out):
    for ipart in range(len(hci)):
        md_idx = tuple(hci[ipart])
        out[md_idx] += field[ipart]
