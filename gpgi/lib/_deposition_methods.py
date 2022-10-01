def _deposit_pic(pcount, hci, pfield, buffer):
    for ipart in range(pcount):
        md_idx = tuple(hci[ipart])
        buffer[md_idx] += pfield[ipart].d
