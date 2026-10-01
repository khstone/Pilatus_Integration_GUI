"""Help text for live mode and event detection. One source for the Help dialogs and for the
hover text on each detected event."""

# What each detector family measures, what it may indicate physically, and its blind spots.
FAMILIES = {
    "area": {
        "name": "area",
        "measures": "Integrated intensity appearing (area+) or disappearing (area−) in a 2θ region, "
                    "beyond what small peak shifts explain. Insensitive to peak width changes.",
        "indicates": "area+: a crystalline phase forming or growing (reaction product, intermediate, "
                     "new polymorph, crystallization from an amorphous or liquid phase, new ordering "
                     "reflections). area−: a phase being consumed, decomposing, melting, becoming "
                     "amorphous, or transforming. Both together: one phase converting into another. "
                     "A loss followed later by a gain at the same positions: a reversible transition "
                     "(e.g. on heating and then cooling).",
        "blind": "Very weak minor phases may stay below threshold.",
        "by_sign": {
            "+": "Intensity appeared: a crystalline phase forming or growing (reaction product, "
                 "intermediate, new polymorph, crystallization from an amorphous or liquid phase, "
                 "new ordering reflections).",
            "-": "Intensity disappeared: a phase being consumed, decomposing, melting, becoming "
                 "amorphous, or transforming.",
            "+-": "Intensity appeared and disappeared together: one phase converting into another "
                  "(a reaction or a phase transition). If the same positions later reverse, the "
                  "transition is reversible.",
        },
    },
    "pointwise": {
        "name": "pointwise",
        "measures": "The same appearing/disappearing comparison on √intensity, point by point.",
        "indicates": "Most sensitive of the families to weak or minor phases. It also responds when "
                     "peaks narrow or broaden (a narrower peak has new height even at constant area): "
                     "pointwise with profile but without area usually means a peak-shape change, "
                     "not a new phase.",
        "blind": "Cannot by itself tell a new weak phase from a peak-shape change.",
        "by_sign": {
            "+": "New intensity point by point: a weak phase forming, or peaks sharpening.",
            "-": "Intensity lost point by point: a weak phase disappearing, or peaks broadening.",
            "+-": "Intensity both gained and lost point by point: a phase change, or peaks "
                  "changing shape or position.",
        },
    },
    "profile": {
        "name": "profile",
        "measures": "Change in peak sharpness at roughly constant integrated intensity.",
        "indicates": "Sharpening: crystallite growth, annealing of strain or defects, sintering, or "
                     "the finest crystallites of a phase being consumed first. Broadening: increasing "
                     "strain or defects, decreasing crystallite size, onset of disorder, or a "
                     "reflection beginning to split before the split is resolved (e.g. symmetry "
                     "lowering).",
        "blind": "Says nothing about which phase changed; look at which reflections changed.",
    },
    "slope": {
        "name": "slope",
        "measures": "How fast the whole pattern is moving away from the first scan.",
        "indicates": "Gradual changes: slow reactions, steadily changing phase fractions, continuous "
                     "transitions. Also responds to changes in heating rate, because thermal "
                     "expansion is continuous.",
        "blind": "Poor at timing abrupt events precisely.",
    },
    "pearson": {
        "name": "pearson",
        "measures": "Abrupt change of the whole pattern between consecutive scans (1 − Pearson r).",
        "indicates": "Fast events. Also the start or end of a hold or of cooling, when thermal "
                     "expansion starts or stops.",
        "blind": "Dominated by the strongest reflections: misses changes in weak phases.",
    },
}

CONFIDENCE = {
    "high": "Two or more independent detector families agree: the pattern changed in more than "
            "one way. Most likely a real transformation. On the validation data every confirmed "
            "transformation was high confidence.",
    "low": "Only one family responded. It may be real (a subtle or gradual change that only one "
           "family sees) or a false alarm (sample movement, beam or intensity fluctuations, "
           "crowded peaks near the edge of the 2θ range, noise early in a live run). Look at the "
           "patterns before deciding.",
}

DIRECTION = {
    "sharpening": "Peaks narrowed at roughly constant area (see profile).",
    "broadening": "Peaks broadened at roughly constant area (see profile).",
}


def _family_rows():
    rows = []
    for f in FAMILIES.values():
        rows.append(f"<tr><td><b>{f['name']}</b></td><td>{f['measures']}</td>"
                    f"<td>{f['indicates']}</td><td>{f['blind']}</td></tr>")
    return "\n".join(rows)


EVENTS_HTML = f"""
<h2>Understanding detected events</h2>
<p>An event is a scan where the diffraction pattern changed in a way that smooth thermal
expansion does not explain. Detection does not know which phases are present: it flags
<i>where</i> something changed. You decide <i>what</i> changed.</p>

<h3>Reading an event</h3>
<p><code>scan 96 [high] area+pearson+pointwise+profile+slope, sharpening, 905 °C</code></p>
<ul>
<li><b>scan 96</b>: where the change is centred. In live mode it is reported about 6 scans
later, once the detectors can confirm it.</li>
<li><b>[high] / [low]</b>: confidence (below).</li>
<li><b>area+pearson+…</b>: the detector families that responded (below).</li>
<li><b>sharpening / broadening</b>: shown when peak widths changed.</li>
<li><b>905 °C</b>: temperature at the event, when the furnace temperature is available.</li>
</ul>
<p>On the waterfall, high-confidence events are solid lines and low-confidence events dotted.</p>

<h3>Confidence</h3>
<p><b>[high]</b>: {CONFIDENCE['high']}</p>
<p><b>[low]</b>: {CONFIDENCE['low']}</p>

<h3>Detector families</h3>
<p>Each pattern is compared with the pattern 3 scans earlier, allowing peaks to shift by up to
±0.05° 2θ so that thermal expansion and contraction are not events.</p>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Family</th><th>Measures</th><th>May indicate</th><th>Blind spots</th></tr>
{_family_rows()}
</table>

<h3>What is not detected</h3>
<ul>
<li>A phase that is present but not changing. "No events during a hold" means nothing changed
beyond the thresholds, not that a reaction is complete.</li>
<li>Changes smaller than the noise (roughly a percent of the pattern's intensity).</li>
<li>Smooth peak shifts (thermal expansion, gradual composition change in a solid solution) are
ignored by design. A sudden lattice change larger than 0.05° can appear as area or pointwise
events.</li>
</ul>

<h3>Using events</h3>
<p>Events are suggestions for region boundaries in a sequential refinement. Before relying on
one, compare the patterns just before and just after it.</p>
<p><b>Live vs loaded data.</b> Live mode compares each pattern with the history so far, so it
reports more low-confidence events early in a run. <i>Analysis &gt; Detect Events in Selected
Data</i> uses the whole run as its baseline and usually gives fewer false alarms.</p>
"""

LIVE_HTML = """
<h2>Live mode</h2>
<p>Integrates each scan automatically as soon as it is complete and shows a live waterfall
(intensity vs 2θ and scan number). No clicks are needed during the experiment.</p>
<ol>
<li>Load the calibration file, SPEC file, image path, and output path as usual.</li>
<li>Tick <b>Live integration</b>. Leave <b>Start at</b> blank to begin with the scan in progress
(or the next one), or enter a scan number to also integrate earlier scans already collected.</li>
<li>Each finished scan is integrated, written as <code>.xye</code> to the output path, added to
the data list, and drawn on the waterfall. Untick to stop.</li>
</ol>
<p>Manual <b>Integrate</b> is disabled while live mode runs.</p>

<h3>When a scan is integrated</h3>
<p>When its SPEC block has all its points (for <code>ascan</code>: intervals + 1; other scan
types: when the next scan starts) and every image is present at full size with its
<code>.pdi</code> file. A half-written image is never integrated, and no scan is integrated
twice. Scans that never get images (e.g. alignment scans) are skipped after 30 s.</p>

<h3>Detect events</h3>
<p>With <b>Detect events</b> ticked (needs the <i>insitu-seg</i> package), transformations are
flagged a few scans after they happen and listed in the <b>Events</b> tab. See
<i>Help &gt; Understanding Detected Events</i>. Hover over an event for an explanation of it.</p>
"""


def event_explanation(ev) -> str:
    """Hover text for one event (rich text)."""
    lines = [f"<b>Scan {ev.scan}</b>, {ev.confidence} confidence"
             + (f", {ev.T_C:.0f} °C" if getattr(ev, "T_C", None) is not None else "")]
    lines.append(CONFIDENCE.get(ev.confidence, ""))
    for fam in ev.families:
        f = FAMILIES.get(fam)
        if not f:
            continue
        signs = "".join(sorted({c[-1] for c in ev.classes if c.startswith(fam) and c[-1] in "+-"}))
        tag = {"+": " (appearing)", "-": " (disappearing)", "+-": " (appearing and disappearing)"}.get(signs, "")
        meaning = f.get("by_sign", {}).get(signs, f["indicates"])
        lines.append(f"<b>{fam}{tag}</b>: {meaning}")
    if getattr(ev, "profile_direction", None):
        lines.append(f"<b>{ev.profile_direction}</b>: {DIRECTION[ev.profile_direction]}")
    lines.append("<i>A suggestion: compare the patterns before and after this scan.</i>")
    return "<qt>" + "<br><br>".join(l for l in lines if l) + "</qt>"
