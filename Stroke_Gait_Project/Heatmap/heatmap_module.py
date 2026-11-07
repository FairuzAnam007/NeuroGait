from pathlib import Path
import csv
import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
HEATMAP_DIR = PROJECT_ROOT / "data" / "heatmaps" / "valid"
ASSIGNMENTS_FILE = PROJECT_ROOT / "data" / "heatmaps" / "reference_assignments.csv"


def assign_reference(participant_key):

    images = sorted(HEATMAP_DIR.glob('*.png'))
    if not images:
        return None, "No heatmaps found. Extract the supplied ZIP in the inner project folder."
    assignments = {}
    if ASSIGNMENTS_FILE.is_file():
        with ASSIGNMENTS_FILE.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                assignments[row['participant_key']] = row['image_file']
    if participant_key in assignments:
        p=HEATMAP_DIR / assignments[participant_key]
        if p.is_file() and p.stat().st_size:
            return p, None
        return None, f"Previously assigned image missing for {participant_key}."
    used=set(assignments.values())
    choices=[p for p in images if p.name not in used]
    if not choices:
        return None, ("All valid reference images are already assigned. "
                      "239 unique reference images require at least 239 valid files; only 202 are available.")
    chosen=choices[0]
    ASSIGNMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ASSIGNMENTS_FILE.open('a', newline='', encoding='utf-8') as f:
        writer=csv.writer(f)
        if f.tell()==0: writer.writerow(['participant_key','image_file'])
        writer.writerow([participant_key,chosen.name])
    return chosen, None


def _part_features(rgb, x_start, x_end):

    rgb=rgb[:,x_start:x_end,:]
    mx=rgb.max(axis=2).astype(np.int16)
    mn=rgb.min(axis=2).astype(np.int16)
    mask=(mx>=60) & ((mx-mn)>=45)
    column_counts=mask.sum(axis=0)
    active=np.flatnonzero(column_counts >= 3)
    if len(active) == 0:
        return None
    breaks=np.flatnonzero(np.diff(active) > 4) + 1
    groups=np.split(active, breaks)
    main=max(groups, key=lambda g: int(column_counts[g].sum()))
    mask[:,:int(main[0])]=False
    mask[:,int(main[-1])+1:]=False
    yy,xx=np.where(mask)
    if len(yy)<120:
        return None
    y0,y1=int(yy.min()),int(yy.max())
    x0,x1=int(xx.min()),int(xx.max())
    height=max(1,y1-y0+1)
    width=max(1,x1-x0+1)
    crop=mask[y0:y1+1,x0:x1+1]
    ylo=int(round(.19*height)); yhi=height
    middle_lo=int(round(.43*height)); middle_hi=int(round(.68*height))
    denominator=int(crop[ylo:yhi].sum())
    midcount=int(crop[middle_lo:middle_hi].sum())
    midshare=midcount/denominator if denominator else 0
    def median_width(a,b):
        vals=[]
        for row in crop[int(a*height):int(b*height)]:
            xs=np.flatnonzero(row)
            if len(xs)>=2: vals.append(xs[-1]-xs[0]+1)
        return float(np.median(vals)) if vals else float('nan')
    return dict(contact_pixels=int(mask.sum()),length_px=height,width_px=width,
                forefoot_width_px=median_width(.19,.44),
                heel_width_px=median_width(.77,.96),
                midfoot_contact_share=midshare)


def estimate_image_indicators(image):
    rgb=np.asarray(image.convert('RGB'))
    w=rgb.shape[1]
    left=_part_features(rgb,int(w*.02),int(w*.48))
    right=_part_features(rgb,int(w*.52),int(w*.98))
    if not left or not right: return None
    la,ra=left['contact_pixels'],right['contact_pixels']
    diff=100*abs(la-ra)/max(1,la+ra)
    return left,right,diff


def show_reference_heatmap(condition, participant_id):
    st.subheader('Plantar heatmap and foot-contact profile')
    st.caption('Independent reference image dataset — not pressure measured from this participant.')
    participant_id=str(participant_id or '').strip()
    if not participant_id:
        st.info('Enter the participant ID above before analysing to assign a consistent reference image.')
        return
    key=f'{condition}:{participant_id}'
    path,message=assign_reference(key)
    if message:
        st.warning(message)
        return
    image=Image.open(path).convert('RGB')
    col_image,col_data=st.columns([1.1,1])
    with col_image:
        st.image(image, caption=f'Reference heatmap: {path.name}', width='stretch')
    with col_data:
        st.markdown('**Image-derived contact indicators**')
        features=estimate_image_indicators(image)
        if features:
            left,right,asym=features
            a,b=st.columns(2)
            a.metric('Left midfoot contact share',f"{left['midfoot_contact_share']*100:.1f}%")
            b.metric('Right midfoot contact share',f"{right['midfoot_contact_share']*100:.1f}%")
            st.metric('Left/right contact-area difference',f'{asym:.1f}%')
            import pandas as pd
            st.dataframe(pd.DataFrame([
                {'Feature':'Footprint length (pixels)','Left':left['length_px'],'Right':right['length_px']},
                {'Feature':'Forefoot contact width (pixels)','Left':round(left['forefoot_width_px'],1),'Right':round(right['forefoot_width_px'],1)},
                {'Feature':'Heel contact width (pixels)','Left':round(left['heel_width_px'],1),'Right':round(right['heel_width_px'],1)},
                {'Feature':'Contact pixels','Left':left['contact_pixels'],'Right':right['contact_pixels']},
            ]), hide_index=True, width='stretch')
        else:
            st.warning('This scan could not be segmented reliably; no contact measurements displayed.')
        st.write('**Calibrated plantar pressure (kPa):** Not available')
        st.write('**Plantar temperature (°C):** Not available')
        st.write('**Arch height (mm):** Not available')
        st.caption('Contact indicators are approximate pixel measurements. They are not clinical arch measurements, measured pressure or a DFU diagnosis.')
    st.info('This heatmap was assigned as a visual reference and is not from the uploaded gait participant. It is not used by the Stroke or Parkinson’s CNN-LSTM prediction.')
