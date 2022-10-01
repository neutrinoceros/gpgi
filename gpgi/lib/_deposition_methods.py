def _deposit_pic(pcount, hci, pfield, out):
    for ipart in range(pcount):
        md_idx = tuple(hci[ipart])
        out[md_idx] += pfield[ipart]
