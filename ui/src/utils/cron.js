/**
 * Cron expression → Human-readable (German).
 */
export function cronToHuman(expr) {
  if (!expr || typeof expr !== "string") return "";
  const parts = expr.trim().split(/\s+/);
  if (parts.length < 5) return "Ungültiger Ausdruck";
  const [min, hour, dom, mon, dow] = parts;
  const dayNames = { 0: "So", 1: "Mo", 2: "Di", 3: "Mi", 4: "Do", 5: "Fr", 6: "Sa", 7: "So" };
  const monNames = { 1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez" };
  let time = "";
  if (hour !== "*" && min !== "*") time = `um ${hour.padStart(2, "0")}:${min.padStart(2, "0")}`;
  else if (hour !== "*") time = `zur Stunde ${hour}`;
  else time = "jede Minute";
  let days = "";
  if (dow !== "*") {
    const ranges = dow.split(",").map(d => {
      if (d.includes("-")) {
        const [a, b] = d.split("-");
        return `${dayNames[a] || a}–${dayNames[b] || b}`;
      }
      return dayNames[d] || d;
    });
    days = ranges.join(", ");
  }
  let months = "";
  if (mon !== "*") {
    months = mon.split(",").map(m => monNames[m] || m).join(", ");
  }
  let domStr = "";
  if (dom !== "*") domStr = `am ${dom}.`;
  let result = time;
  if (days) result += ` (${days})`;
  if (domStr) result += ` ${domStr}`;
  if (months) result += ` in ${months}`;
  return result;
}
