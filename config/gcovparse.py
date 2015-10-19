version = VERSION = __version__ = "0.0.4"


def gcovparse(combined):
    # clean and strip lines
    assert ':Source:' in combined, 'gcov file is missing ":Source:" line(s)'
    files = filter(lambda f: f != '', combined.strip().split("0:Source:"))
    reports = map(_part, files[1:])
    return reports


def _part(chunk):
    report = {
        "file": chunk.split('\n', 1)[0],
        "lines": []
    }
    map(lambda l: _line(l, report), chunk.strip().split('\n')[1:])

    return report


def _line(l, report):
    line = l.split(':', 2)
    if len(line) == 3:
        hit, line, data = tuple(line)
        if '#' in hit and data.strip() == '}':
            # ignore lines #####:   33:}
            return None
        elif '-' in hit:
            return None
        elif int(line) > 0:
            report['lines'].append(dict(line=int(line.strip()), hit=0 if '#' in hit or '=' in hit else int(hit.strip())))
