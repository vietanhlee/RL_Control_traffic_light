import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Bike,
  Car,
  Clock,
  Gauge,
  Layers3,
  Server,
  Settings2,
  Sparkles,
  TriangleAlert,
} from "lucide-react";

export default function App() {
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const backgroundNeedsRedraw = useRef(true);

  const [metrics, setMetrics] = useState(null);
  const [canvasData, setCanvasData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [selectedIntersection, setSelectedIntersection] = useState(0);
  const [isSwitching, setIsSwitching] = useState(false);
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8011";

  // 1. Fetch Network Topology
  useEffect(() => {
    const fetchNetwork = async () => {
      try {
        const res = await fetch(`${apiBase}/api/v1/network`);
        if (res.ok) {
          const data = await res.json();
          setNetworkData(data);
          // Set initial intersection to the first available node
          if (data.nodes && Object.keys(data.nodes).length > 0) {
            setSelectedIntersection(Number(Object.keys(data.nodes)[0]));
          }
        }
      } catch (err) {
        console.error("Failed to fetch network topology", err);
      }
    };
    fetchNetwork();
  }, [apiBase]);

  // 2. WebSocket for live rendering
  useEffect(() => {
    const ws = new WebSocket(
      `${apiBase.replace(/^http/, "ws")}/ws/simulation/render`,
    );
    let lastRenderTime = 0;
    const renderThrottleMs = 300; // Render xe ở tần số thấp ~3.3 FPS để chống lag bản đồ

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Cập nhật các chỉ số ở tần số gốc của WebSocket
      setMetrics(data);

      // Cập nhật dữ liệu vẽ Canvas ở tần số thấp
      const now = Date.now();
      if (now - lastRenderTime >= renderThrottleMs) {
        setCanvasData({
          time_s: data.time_s,
          vehicles: data.vehicles,
          lights: data.lights,
        });
        lastRenderTime = now;
      }
    };

    return () => ws.close();
  }, [apiBase]);

  // Trigger redraw of static background when network topology changes
  useEffect(() => {
    backgroundNeedsRedraw.current = true;
  }, [networkData]);

  // 4. Force Phase Switch Action
  const handleForceSwitch = async () => {
    setIsSwitching(true);
    try {
      await fetch(`${apiBase}/api/v1/action/${selectedIntersection}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: 1 }),
      });
    } catch (err) {
      console.error("Action failed", err);
    }
    setTimeout(() => setIsSwitching(false), 500);
  };

  const getCanvasTransform = (width, height, networkData) => {
    if (!networkData || !networkData.nodes) return null;
    const nodesArray = Object.values(networkData.nodes);
    let minX = 0, maxX = 900, minY = 0, maxY = 700;
    if (nodesArray.length > 0) {
      minX = Math.min(...nodesArray.map((n) => n.x));
      maxX = Math.max(...nodesArray.map((n) => n.x));
      minY = Math.min(...nodesArray.map((n) => n.y));
      maxY = Math.max(...nodesArray.map((n) => n.y));
    }
    const mapWidth = maxX - minX || 1;
    const mapHeight = maxY - minY || 1;
    const padding = 50;
    const scaleX = (width - padding * 2) / mapWidth;
    const scaleY = (height - padding * 2) / mapHeight;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (width - mapWidth * scale) / 2 - minX * scale;
    const offsetY = (height - mapHeight * scale) / 2 - minY * scale;

    const mapX = (x) => x * scale + offsetX;
    const mapY = (y) => y * scale + offsetY;
    const laneWidth = 10 * scale; 
    const carScale = Math.max(0.4, scale / 0.95);
    
    return { mapX, mapY, scale, offsetX, offsetY, laneWidth, carScale };
  };

  const handleCanvasClick = (e) => {
    if (!canvasRef.current || !networkData) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    const transform = getCanvasTransform(canvasRef.current.width, canvasRef.current.height, networkData);
    if (!transform) return;
    const { mapX, mapY, laneWidth } = transform;

    let closestNode = null;
    let minDistance = Infinity;

    Object.entries(networkData.nodes).forEach(([nodeId, pos]) => {
      const cx = mapX(pos.x);
      const cy = mapY(pos.y);
      const dist = Math.hypot(clickX - cx, clickY - cy);
      
      const connectedEdges = networkData.edges.filter(
          (edge) => edge.start === Number(nodeId) || edge.end === Number(nodeId),
      );
      const maxLanes = connectedEdges.reduce((m, edge) => Math.max(m, edge.lanes), 1);
      const R = maxLanes * laneWidth * 1.15;
      
      if (dist < R + 30 && dist < minDistance) {
        minDistance = dist;
        closestNode = Number(nodeId);
      }
    });

    if (closestNode !== null) {
      setSelectedIntersection(closestNode);
    }
  };

  // 5. Canvas Renderer
  useEffect(() => {
    if (!canvasRef.current || !canvasData || !networkData) return;
    const ctx = canvasRef.current.getContext("2d");
    const width = canvasRef.current.width;
    const height = canvasRef.current.height;

    const transform = getCanvasTransform(width, height, networkData);
    if (!transform) return;
    const { mapX, mapY, laneWidth, carScale } = transform;


    // Khởi tạo hoặc thay đổi kích thước offscreen canvas đệm nền tĩnh
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
      backgroundNeedsRedraw.current = true;
    }

    if (
      offscreenCanvasRef.current.width !== width ||
      offscreenCanvasRef.current.height !== height
    ) {
      offscreenCanvasRef.current.width = width;
      offscreenCanvasRef.current.height = height;
      backgroundNeedsRedraw.current = true;
    }

    // Chỉ vẽ lại nền tĩnh (đường sá, dải phân làn, thảm cỏ, vạch kẻ...) khi thực sự cần thiết
    if (backgroundNeedsRedraw.current) {
      const oCtx = offscreenCanvasRef.current.getContext("2d");

      // Xoá nền màu Slate-950 tối
      oCtx.fillStyle = "#030712";
      oCtx.fillRect(0, 0, width, height);

      // Vẽ toàn bộ các tuyến đường (Roads/Edges) lên offscreen canvas
      networkData.edges.forEach((edge) => {
        const startNode = networkData.nodes[edge.start];
        const endNode = networkData.nodes[edge.end];
        if (!startNode || !endNode) return;

        const sx = mapX(startNode.x);
        const sy = mapY(startNode.y);
        const ex = mapX(endNode.x);
        const ey = mapY(endNode.y);
        const roadWidth = edge.lanes * laneWidth * 2;
        const isMultiLane = edge.lanes > 2;

        // 1. Nền nhựa đường tối
        oCtx.strokeStyle = "#0f172a";
        oCtx.lineWidth = roadWidth;
        oCtx.lineCap = "butt";
        oCtx.beginPath();
        oCtx.moveTo(sx, sy);
        oCtx.lineTo(ex, ey);
        oCtx.stroke();

        const dx = ex - sx;
        const dy = ey - sy;
        const L = Math.hypot(dx, dy);
        if (L > 1e-6) {
          const nx = dy / L;
          const ny = -dx / L;

          // 2. Lề đường ngoài (vạch liền)
          oCtx.strokeStyle = "#64748b";
          oCtx.lineWidth = 1.5;
          oCtx.setLineDash([]);

          // Mép phải
          oCtx.beginPath();
          oCtx.moveTo(sx + nx * (roadWidth / 2), sy + ny * (roadWidth / 2));
          oCtx.lineTo(ex + nx * (roadWidth / 2), ey + ny * (roadWidth / 2));
          oCtx.stroke();

          // Mép trái
          oCtx.beginPath();
          oCtx.moveTo(sx - nx * (roadWidth / 2), sy - ny * (roadWidth / 2));
          oCtx.lineTo(ex - nx * (roadWidth / 2), ey - ny * (roadWidth / 2));
          oCtx.stroke();

          // 3. Vạch đứt phân làn trong từng chiều
          oCtx.strokeStyle = "#475569";
          oCtx.lineWidth = 1.0;
          oCtx.setLineDash([5, 10]);

          // Hướng thuận (offset dương từ tim)
          for (let i = 1; i < edge.lanes; i++) {
            const offset = i * laneWidth;
            oCtx.beginPath();
            oCtx.moveTo(sx + nx * offset, sy + ny * offset);
            oCtx.lineTo(ex + nx * offset, ey + ny * offset);
            oCtx.stroke();
          }

          // Hướng nghịch (offset âm từ tim)
          for (let i = 1; i < edge.lanes; i++) {
            const offset = i * laneWidth;
            oCtx.beginPath();
            oCtx.moveTo(sx - nx * offset, sy - ny * offset);
            oCtx.lineTo(ex - nx * offset, ey - ny * offset);
            oCtx.stroke();
          }
          oCtx.setLineDash([]);

          // 4. Dải phân cách giữa đường (tim đường)
          if (isMultiLane) {
            const barrierWidth = laneWidth * 0.9;

            oCtx.strokeStyle = "#374151";
            oCtx.lineWidth = barrierWidth;
            oCtx.setLineDash([]);
            oCtx.beginPath();
            oCtx.moveTo(sx, sy);
            oCtx.lineTo(ex, ey);
            oCtx.stroke();

            oCtx.strokeStyle = "#166534";
            oCtx.lineWidth = barrierWidth * 0.65;
            oCtx.beginPath();
            oCtx.moveTo(sx, sy);
            oCtx.lineTo(ex, ey);
            oCtx.stroke();

             const halfB = barrierWidth / 2;
            oCtx.strokeStyle = "#94a3b8"; // Tăng độ sáng viền dải phân cách
            oCtx.lineWidth = 1.0;
            oCtx.beginPath();
            oCtx.moveTo(sx + nx * halfB, sy + ny * halfB);
            oCtx.lineTo(ex + nx * halfB, ey + ny * halfB);
            oCtx.stroke();
            oCtx.beginPath();
            oCtx.moveTo(sx - nx * halfB, sy - ny * halfB);
            oCtx.lineTo(ex - nx * halfB, ey - ny * halfB);
            oCtx.stroke();
          } else {
            oCtx.strokeStyle = "#eab308";
            oCtx.lineWidth = 1.5;
            oCtx.setLineDash([8, 8]);
            oCtx.beginPath();
            oCtx.moveTo(sx, sy);
            oCtx.lineTo(ex, ey);
            oCtx.stroke();
            oCtx.setLineDash([]);
          }
        }
      });

      // Vẽ toàn bộ các nút giao (Intersections) lên offscreen canvas
      Object.entries(networkData.nodes).forEach(([nodeId, pos]) => {
        const cx = mapX(pos.x);
        const cy = mapY(pos.y);

        const connectedEdges = networkData.edges.filter(
          (e) => e.start === Number(nodeId) || e.end === Number(nodeId),
        );
        const maxLanes = connectedEdges.reduce(
          (m, e) => Math.max(m, e.lanes),
          1,
        );
        const R = maxLanes * laneWidth * 1.15;

        oCtx.save();
        oCtx.translate(cx, cy);

        // Asphalt nền (Đồng bộ với mặt đường tối hơn)
        oCtx.fillStyle = "#0f172a";
        oCtx.beginPath();
        oCtx.arc(0, 0, R, 0, Math.PI * 2);
        oCtx.fill();

        // Kẻ vạch dừng xe & Zebra crossing
        connectedEdges.forEach((edge) => {
          const neighborId =
            edge.start === Number(nodeId) ? edge.end : edge.start;
          const neighbor = networkData.nodes[neighborId];
          if (!neighbor) return;

          const dx = mapX(neighbor.x) - cx;
          const dy = mapY(neighbor.y) - cy;
          const dL = Math.hypot(dx, dy);
          if (dL < 1e-6) return;
          const ux = dx / dL;
          const uy = dy / dL;
          const px = uy;
          const py = -ux;

          const rHW = edge.lanes * laneWidth;
          const stopDist = R + 1.5;
          oCtx.strokeStyle = "rgba(255,255,255,0.95)"; // Sáng rõ vạch dừng
          oCtx.lineWidth = 2.0;
          oCtx.setLineDash([]);
          oCtx.beginPath();
          oCtx.moveTo(
            ux * stopDist + px * rHW * 0.95,
            uy * stopDist + py * rHW * 0.95,
          );
          oCtx.lineTo(
            ux * stopDist - px * rHW * 0.95,
            uy * stopDist - py * rHW * 0.95,
          );
          oCtx.stroke();

          if (edge.lanes >= 2) {
            const zEnd = R * 0.95;
            const zStart = R * 0.72;
            const numStripes = edge.lanes + 1;
            const halfSpan = rHW * 0.88;

            oCtx.fillStyle = "rgba(255,255,255,0.25)"; // Sáng rõ vạch đi bộ
            for (let s = 0; s < numStripes; s++) {
              if (s % 2 === 0) continue;
              const t0 = (s / numStripes) * 2 * halfSpan - halfSpan;
              const t1 = ((s + 1) / numStripes) * 2 * halfSpan - halfSpan;
              oCtx.beginPath();
              oCtx.moveTo(ux * zStart + px * t0, uy * zStart + py * t0);
              oCtx.lineTo(ux * zEnd + px * t0, uy * zEnd + py * t0);
              oCtx.lineTo(ux * zEnd + px * t1, uy * zEnd + py * t1);
              oCtx.lineTo(ux * zStart + px * t1, uy * zStart + py * t1);
              oCtx.closePath();
              oCtx.fill();
            }
          }
        });

        // Specular highlight nhựa ướt
        const specGrad = oCtx.createRadialGradient(
          -R * 0.2,
          -R * 0.2,
          0,
          0,
          0,
          R,
        );
        specGrad.addColorStop(0, "rgba(100,116,139,0.28)");
        specGrad.addColorStop(0.5, "rgba(51,65,85,0.10)");
        specGrad.addColorStop(1, "rgba(15,23,42,0)");
        oCtx.fillStyle = specGrad;
        oCtx.beginPath();
        oCtx.arc(0, 0, R, 0, Math.PI * 2);
        oCtx.fill();

        // Viền kerb
        oCtx.strokeStyle = "rgba(148,163,184,0.75)"; // Sáng rõ viền bồn ngã tư
        oCtx.lineWidth = 0.8;
        oCtx.setLineDash([]);
        oCtx.beginPath();
        oCtx.arc(0, 0, R, 0, Math.PI * 2);
        oCtx.stroke();

        // Chấm tâm node marker
        oCtx.fillStyle = "rgba(148,163,184,0.6)";
        oCtx.beginPath();
        oCtx.arc(0, 0, 2.2, 0, Math.PI * 2);
        oCtx.fill();

        // Label số nút giao
        const fontSize = Math.max(7, laneWidth * 0.6);
        oCtx.fillStyle = "rgba(241,245,249,0.9)"; // Chữ số màu trắng tinh rõ nét
        oCtx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
        oCtx.textAlign = "center";
        oCtx.textBaseline = "middle";
        oCtx.fillText(nodeId, R * 0.52, -R * 0.58);

        oCtx.restore();
      });

      backgroundNeedsRedraw.current = false;
    }

    // BƯỚC 1: Vẽ ảnh đệm background đã được cache sang canvas chính
    ctx.drawImage(offscreenCanvasRef.current, 0, 0);

    // BƯỚC 2: Vẽ hiệu ứng Glow động và Highlight nhấp nháy cho giao lộ được chọn lên canvas chính
    const activeNode = networkData.nodes[selectedIntersection];
    if (activeNode) {
      const cx = mapX(activeNode.x);
      const cy = mapY(activeNode.y);
      const connectedEdges = networkData.edges.filter(
        (e) =>
          e.start === selectedIntersection || e.end === selectedIntersection,
      );
      const maxLanes = connectedEdges.reduce((m, e) => Math.max(m, e.lanes), 1);
      const R = maxLanes * laneWidth * 1.15;

      ctx.save();
      ctx.translate(cx, cy);

      const t = Date.now() / 350;
      const pulse = (Math.sin(t) + 1) / 2;

      ctx.strokeStyle = `rgba(56,189,248,${0.45 + pulse * 0.5})`;
      ctx.lineWidth = 1.8 + pulse * 1.8;
      ctx.shadowBlur = 16 + pulse * 12;
      ctx.shadowColor = "#38bdf8";
      ctx.beginPath();
      ctx.arc(0, 0, R + 5 + pulse * 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      const hlGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 0.85);
      hlGrad.addColorStop(0, `rgba(56,189,248,${0.1 + pulse * 0.08})`);
      hlGrad.addColorStop(1, "rgba(56,189,248,0)");
      ctx.fillStyle = hlGrad;
      ctx.beginPath();
      ctx.arc(0, 0, R * 0.85, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    }

    // BƯỚC 3: Vẽ Đèn tín hiệu giao thông (Traffic Lights) - cập nhật động mỗi frame
    canvasData.lights?.forEach((light) => {
      const node = networkData.nodes[light.intersection];
      const incomingNode = networkData.nodes[light.incoming];
      if (!node || !incomingNode) return;
      const x = mapX(node.x);
      const y = mapY(node.y);
      const sx = mapX(incomingNode.x);
      const sy = mapY(incomingNode.y);

      const dx = x - sx;
      const dy = y - sy;
      const L = Math.hypot(dx, dy);

      if (L > 1e-6) {
        const ux = dx / L;
        const uy = dy / L;
        const nx = uy;
        const ny = -ux;

        const edge = networkData.edges.find(
          (e) =>
            (e.start === light.intersection && e.end === light.incoming) ||
            (e.start === light.incoming && e.end === light.intersection),
        );
        const lanes = edge ? edge.lanes : 1;
        const stopLineDistance = 3 * laneWidth + 12 * carScale;
        const sideOffset = lanes * laneWidth * 0.85;

        // Vẽ overlay dải màu nền trên mặt đường (chiều dài = 1/3 đoạn đường)
        const overlayLength = L / 3;
        const startX = x - ux * stopLineDistance;
        const startY = y - uy * stopLineDistance;
        const endX = startX - ux * overlayLength;
        const endY = startY - uy * overlayLength;
        const halfRoadCenterOffset = (lanes * laneWidth) / 2;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(startX + nx * halfRoadCenterOffset, startY + ny * halfRoadCenterOffset);
        ctx.lineTo(endX + nx * halfRoadCenterOffset, endY + ny * halfRoadCenterOffset);
        
        const overlayGrad = ctx.createLinearGradient(startX, startY, endX, endY);
        const baseColor = light.color === "RED" ? "239, 68, 68" : light.color === "YELLOW" ? "245, 158, 11" : "34, 197, 94";
        // Tăng opacity tối đa ở điểm bắt đầu (0.95) và giữ độ sáng mịn hơn (0.4 ở giữa) trước khi chuyển về 0.0
        overlayGrad.addColorStop(0, `rgba(${baseColor}, 0.95)`);
        overlayGrad.addColorStop(0.4, `rgba(${baseColor}, 0.4)`);
        overlayGrad.addColorStop(1, `rgba(${baseColor}, 0.0)`);
        
        ctx.strokeStyle = overlayGrad;
        ctx.lineWidth = lanes * laneWidth * 1.0; // Mở rộng phủ kín toàn bộ chiều rộng làn đường
        ctx.lineCap = "butt";
        ctx.stroke();
        ctx.restore();
      }
    });

    // BƯỚC 4: Vẽ Xe cộ (Vehicles) - cập nhật động mỗi frame
    canvasData.vehicles.forEach((v) => {
      const x = mapX(v.x);
      const y = mapY(v.y);
      const angle = v.angle || 0;

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      ctx.scale(carScale, carScale);

      if (v.type === "motorcycle") {
        ctx.fillStyle = "#1e293b";
        ctx.beginPath();
        ctx.ellipse(-3.8, 0, 1.6, 1.0, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#475569";
        ctx.beginPath();
        ctx.ellipse(-3.8, 0, 1.0, 0.6, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#1e293b";
        ctx.beginPath();
        ctx.ellipse(3.6, 0, 1.4, 0.85, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#475569";
        ctx.beginPath();
        ctx.ellipse(3.6, 0, 0.85, 0.5, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(-2.8, 0);
        ctx.lineTo(2.8, 0);
        ctx.stroke();

        ctx.shadowBlur = 6;
        ctx.shadowColor = "#38bdf8";
        ctx.fillStyle = "#0ea5e9";
        ctx.beginPath();
        ctx.roundRect(-2.6, -1.1, 4.5, 2.2, 1.0);
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.fillStyle = "rgba(186,230,253,0.5)";
        ctx.beginPath();
        ctx.roundRect(-2.4, -0.35, 4.0, 0.7, 0.3);
        ctx.fill();

        ctx.fillStyle = "#0369a1";
        ctx.beginPath();
        ctx.roundRect(1.5, -0.9, 1.5, 1.8, 0.4);
        ctx.fill();

        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(2.2, -1.5);
        ctx.lineTo(2.2, 1.5);
        ctx.stroke();

        ctx.fillStyle = "#f1f5f9";
        ctx.beginPath();
        ctx.ellipse(-0.6, 0, 1.05, 0.75, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#f97316";
        ctx.shadowBlur = 3;
        ctx.shadowColor = "#f97316";
        ctx.beginPath();
        ctx.arc(-0.6, 0, 0.75, Math.PI, 0);
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.fillStyle = "rgba(186,230,253,0.6)";
        ctx.beginPath();
        ctx.roundRect(-1.0, -0.15, 0.8, 0.3, 0.15);
        ctx.fill();
      } else if (v.type === "bus") {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(4, 3.5, 3, 1);
        ctx.fillRect(4, -4.5, 3, 1);
        ctx.fillRect(-6, 3.5, 3, 1);
        ctx.fillRect(-6, -4.5, 3, 1);

        ctx.fillStyle = "#f97316";
        ctx.shadowBlur = 6;
        ctx.shadowColor = "#f97316";
        ctx.beginPath();
        ctx.roundRect(-9, -3.5, 18, 7, 2);
        ctx.fill();

        ctx.fillStyle = "#0b1329";
        ctx.shadowBlur = 0;
        ctx.fillRect(6.5, -3, 1.5, 6);

        ctx.fillStyle = "#1e293b";
        for (let offset = -7; offset <= 3; offset += 3.5) {
          ctx.fillRect(offset, -3.2, 2, 0.6);
          ctx.fillRect(offset, 2.6, 2, 0.6);
        }
      } else {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(2.2, 2.3, 2, 1);
        ctx.fillRect(2, -3.3, 2, 1);
        ctx.fillRect(-4.2, 2.3, 2, 1);
        ctx.fillRect(-4.2, -3.3, 2, 1);

        ctx.fillStyle = "#a78bfa";
        ctx.shadowBlur = 5;
        ctx.shadowColor = "#a78bfa";
        ctx.beginPath();
        ctx.roundRect(-5.5, -2.5, 11, 5, 1.5);
        ctx.fill();

        ctx.fillStyle = "#0b1329";
        ctx.shadowBlur = 0;
        ctx.fillRect(1.5, -2, 1.2, 4);
        ctx.fillRect(-4, -2, 0.8, 4);

        ctx.fillStyle = "#c084fc";
        ctx.fillRect(-2.8, -1.8, 3.8, 3.6);
      }
      ctx.restore();
    });
  }, [canvasData, networkData, selectedIntersection]);

  const selectedMetrics =
    metrics?.metrics?.intersections?.[selectedIntersection];
  const totalQueue =
    selectedMetrics && selectedMetrics.directions
      ? Object.values(selectedMetrics.directions).reduce(
          (acc, curr) => acc + curr.queue_length,
          0,
        )
      : 0;
  const selectedAverageSpeed =
    selectedMetrics && selectedMetrics.directions
      ? Object.values(selectedMetrics.directions).reduce(
          (acc, curr) => acc + curr.avg_speed,
          0,
        ) / Math.max(Object.values(selectedMetrics.directions).length, 1)
      : 0;
  const globalImbalance = metrics?.metrics?.global_imbalance ?? 0;
  const availableNodes = networkData ? Object.keys(networkData.nodes) : [];

  // Lấy rewardMetrics trực tiếp từ WebSocket real-time
  const rewardMetrics = metrics?.reward_metrics?.[selectedIntersection];

  return (
    <div className="min-h-screen bg-[#030712] text-slate-100 selection:bg-cyan-400/30 selection:text-white">
      {/* Giảm đáng kể độ sáng mờ của background để tăng tương phản chữ */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-24 left-[-8rem] h-72 w-72 rounded-full bg-cyan-500/5 blur-3xl" />
        <div className="absolute top-28 right-[-6rem] h-80 w-80 rounded-full bg-fuchsia-500/4 blur-3xl" />
        <div className="absolute bottom-[-8rem] left-1/3 h-96 w-96 rounded-full bg-emerald-500/3 blur-3xl" />
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-[1600px] flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="relative overflow-hidden rounded-2xl border border-slate-800 bg-[#0f172a]/95 shadow-2xl backdrop-blur-md transition-all duration-500 hover:border-cyan-500/30 select-none">
          {/* Viền gradient trên cùng của Header */}
          <div className="absolute top-0 left-0 right-0 h-[1.5px] bg-gradient-to-r from-transparent via-cyan-500/60 to-fuchsia-500/40" />

          <div className="flex flex-col gap-4 px-4 py-2 sm:px-6 sm:flex-row sm:items-center sm:justify-center">
            {/* Cột phải: 4 card chỉ số siêu gọn */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Card 1: Global Reward */}
              <div className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/85 px-3 py-1.5 flex items-center gap-3 transition-all duration-300 hover:border-cyan-500/40 shadow-inner">
                <div>
                  <div className="text-[8px] uppercase tracking-[0.12em] text-slate-400 font-bold">
                    Global Reward
                  </div>
                  <div
                    className={`mt-0.5 text-sm font-bold ${(metrics?.global_reward ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                  >
                    {(metrics?.global_reward ?? 0).toFixed(2)}
                  </div>
                </div>
                <Gauge className="h-3.5 w-3.5 text-cyan-400" />
              </div>

              {/* Card 2: Global Queue */}
              <div className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/85 px-3 py-1.5 flex items-center gap-3 transition-all duration-300 hover:border-rose-500/40 shadow-inner">
                <div>
                  <div className="text-[8px] uppercase tracking-[0.12em] text-slate-400 font-bold">
                    Global Queue
                  </div>
                  <div className="mt-0.5 text-sm font-bold text-rose-400">
                    {metrics?.metrics?.intersections
                      ? Object.values(metrics.metrics.intersections).reduce(
                          (sumInter, inter) =>
                            sumInter +
                            Object.values(inter.directions).reduce(
                              (sumDir, dir) => sumDir + dir.queue_length,
                              0,
                            ),
                          0,
                        )
                      : 0}
                  </div>
                </div>
                <TriangleAlert className="h-3.5 w-3.5 text-rose-400" />
              </div>

              {/* Card 3: Active Vehicles */}
              <div className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/85 px-3 py-1.5 flex items-center gap-3 transition-all duration-300 hover:border-violet-500/40 shadow-inner">
                <div>
                  <div className="text-[8px] uppercase tracking-[0.12em] text-slate-400 font-bold">
                    Active Vehicles
                  </div>
                  <div className="mt-0.5 flex items-baseline gap-1.5">
                    <span className="text-sm font-bold text-violet-400">
                      {metrics?.vehicles?.length ?? 0}
                    </span>
                    <span className="text-[8px] text-slate-400 font-semibold tracking-normal">
                      (
                      {metrics?.vehicles.filter(
                        (v) => v.type === "car" || v.type === "bus",
                      ).length || 0}
                      C ·{" "}
                      {metrics?.vehicles.filter((v) => v.type === "motorcycle")
                        .length || 0}
                      M)
                    </span>
                  </div>
                </div>
                <Car className="h-3.5 w-3.5 text-violet-400" />
              </div>

              {/* Card 4: Global Imbalance */}
              <div className="group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950/85 px-3 py-1.5 flex items-center gap-3 transition-all duration-300 hover:border-amber-500/40 shadow-inner">
                <div>
                  <div className="text-[8px] uppercase tracking-[0.12em] text-slate-400 font-bold">
                    Global Imbalance
                  </div>
                  <div className="mt-0.5 text-sm font-bold text-amber-400">
                    {globalImbalance.toFixed(1)}
                  </div>
                </div>
                <Activity className="h-3.5 w-3.5 text-amber-400" />
              </div>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-12 gap-6">
          <main className="col-span-12 xl:col-span-8">
            <section className="overflow-hidden rounded-[2rem] border border-slate-800 bg-[#0f172a]/60 shadow-2xl">
              <div className="flex flex-col gap-4 border-b border-slate-800 px-5 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
                    <Activity className="h-5 w-5 text-cyan-400" />
                    Live Network Map
                  </h2>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-xs font-semibold text-slate-200">
                  <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-violet-400 shadow-[0_0_10px_rgba(167,139,250,0.8)]" />
                    Car
                  </div>
                  <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                    <span className="h-2.5 w-4 rounded-full bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.8)]" />
                    Bus
                  </div>
                  <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/80 px-3 py-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(56,189,248,0.8)]" />
                    Motorcycle
                  </div>
                </div>
              </div>

              <div className="p-4 sm:p-6">
                <div className="relative overflow-hidden rounded-[1.75rem] border border-slate-800 bg-[#030712] shadow-2xl">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.06),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(168,85,247,0.04),transparent_26%)]" />
                  <div className="relative flex min-h-[640px] items-center justify-center">
                    {/* HUD Overlays */}
                    <div className="absolute left-6 top-6 z-10 flex items-center gap-2 rounded-full bg-slate-950 border border-slate-800 px-3 py-1 text-[9px] font-bold uppercase tracking-wider text-slate-200 shadow-md">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                      Live Topology Feed
                    </div>
                    <div className="absolute right-6 top-6 z-10 flex items-center gap-1.5 rounded-full bg-slate-950 border border-slate-800 px-3 py-1 text-[9px] font-mono text-slate-200 shadow-md">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]" />
                      <span>REC</span>
                    </div>

                    <canvas
                      ref={canvasRef}
                      onClick={handleCanvasClick}
                      width={900}
                      height={700}
                      className="block h-auto w-full max-h-[760px] object-contain cursor-pointer"
                    />
                    {!canvasData && (
                      <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
                        <div className="flex flex-col items-center rounded-3xl border border-slate-800 bg-slate-900/90 px-6 py-5 shadow-2xl">
                          <div className="mb-4 h-10 w-10 animate-spin rounded-full border-4 border-cyan-400 border-t-transparent" />
                          <span className="text-sm font-medium tracking-wide text-cyan-200">
                            Connecting to engine...
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          </main>

          <aside className="col-span-12 xl:col-span-4 flex flex-col gap-4">
            <section className="overflow-hidden rounded-[2rem] border border-slate-800 bg-[#0f172a]/60 shadow-2xl">
              <div className="border-b border-slate-800 px-5 py-3 sm:px-6">
                <h3 className="flex items-center gap-2 text-base font-semibold text-white">
                  <Settings2 className="h-5 w-5 text-cyan-400" />
                  Control Panel
                </h3>
              </div>

              <div className="space-y-3 p-4 sm:p-5">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/90 p-3 shadow-md">
                  <div className="flex items-center justify-between">
                    <label className="block text-[10px] font-bold uppercase tracking-[0.22em] text-slate-300">
                      Intersection
                    </label>
                    <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-[#0b1220] px-3 py-1.5 text-xs text-slate-100">
                      <span className="flex h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)] animate-pulse" />
                      <span className="font-semibold tracking-wide">Node #{selectedIntersection}</span>
                    </div>
                  </div>

                  <button
                    onClick={handleForceSwitch}
                    disabled={isSwitching}
                    className={`relative mt-3 inline-flex w-full items-center justify-center gap-2 overflow-hidden rounded-xl px-4 py-2 text-xs font-bold transition-all duration-300 ${
                      isSwitching
                        ? "cursor-not-allowed border border-slate-800 bg-slate-900/40 text-slate-500"
                        : "bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500 text-slate-950 shadow-[0_4px_20px_rgba(6,182,212,0.3)] hover:-translate-y-0.5 hover:shadow-[0_6px_25px_rgba(6,182,212,0.45)] hover:scale-[1.01] active:scale-[0.99]"
                    }`}
                  >
                    {isSwitching ? "Switching Phase..." : "Force Phase Switch"}
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/90 p-3 shadow-md">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-[9px] font-bold uppercase tracking-[0.22em] text-slate-300">
                        Queue Length
                      </div>
                      <div className="mt-1 text-2xl font-black tracking-[-0.05em] text-rose-400">
                        {totalQueue}
                      </div>
                    </div>
                    <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-2 animate-pulse">
                      <TriangleAlert className="h-4 w-4 text-rose-400" />
                    </div>
                  </div>

                  <div className="mt-3 grid gap-2 rounded-xl border border-slate-800 bg-[#0b1220] p-2 shadow-inner">
                    {selectedMetrics && selectedMetrics.directions ? (
                      Object.entries(selectedMetrics.directions).map(
                        ([incomingId, metrics]) => {
                          const maxQueueForBar = 15;
                          const percent = Math.min(
                            (metrics.queue_length / maxQueueForBar) * 100,
                            100,
                          );
                          return (
                            <div
                              key={incomingId}
                              className="flex flex-col gap-1.5 rounded-lg border border-slate-800 bg-slate-900/60 px-2.5 py-1.5"
                            >
                              <div className="flex items-center justify-between text-[11px]">
                                <div>
                                  <span className="font-bold text-slate-200">
                                    Node #{incomingId}
                                  </span>
                                  <span className="ml-2 text-[9px] text-slate-400">
                                    {metrics.avg_speed.toFixed(1)} m/s
                                  </span>
                                </div>
                                <span
                                  className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold ${metrics.queue_length > 5 ? "bg-rose-500/20 text-rose-400 border border-rose-500/30" : "bg-slate-800 text-slate-300"}`}
                                >
                                  Queue: {metrics.queue_length}
                                </span>
                              </div>
                              <div className="h-1 w-full rounded-full bg-[#030712] overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    metrics.queue_length > 8
                                      ? "bg-gradient-to-r from-rose-500 to-red-600 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                                      : metrics.queue_length > 3
                                        ? "bg-gradient-to-r from-amber-400 to-orange-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]"
                                        : "bg-gradient-to-r from-cyan-400 to-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                                  }`}
                                  style={{ width: `${percent}%` }}
                                />
                              </div>
                            </div>
                          );
                        },
                      )
                    ) : (
                      <div className="text-xs text-slate-500 text-center py-2">
                        No metrics available yet.
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/90 p-3 space-y-3 shadow-md">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="text-[9px] font-bold uppercase tracking-[0.22em] text-slate-300">
                      RL Diagnostics
                    </div>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${
                        (rewardMetrics?.reward ?? 0) >= 0
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                      }`}
                    >
                      Reward:{" "}
                      {rewardMetrics?.reward !== undefined
                        ? rewardMetrics.reward.toFixed(2)
                        : "0.00"}
                    </span>
                  </div>

                  <div className="space-y-2">
                    {/* Queue Penalty */}
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-300">Queue Penalty</span>
                        <span className="font-semibold text-rose-300 font-mono">
                          -
                          {rewardMetrics?.queue_penalty !== undefined
                            ? rewardMetrics.queue_penalty.toFixed(2)
                            : "0.00"}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-[#030712] overflow-hidden">
                        <div
                          className="h-full bg-rose-500 rounded-full transition-all duration-300 shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                          style={{
                            width: `${rewardMetrics?.queue_pct ?? 0}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Imbalance Penalty */}
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-300">
                          Imbalance Penalty
                        </span>
                        <span className="font-semibold text-amber-300 font-mono">
                          -
                          {rewardMetrics?.imbalance_penalty !== undefined
                            ? rewardMetrics.imbalance_penalty.toFixed(2)
                            : "0.00"}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-[#030712] overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full transition-all duration-300 shadow-[0_0_8px_rgba(245,158,11,0.6)]"
                          style={{
                            width: `${rewardMetrics?.imbalance_pct ?? 0}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Red Pressure Penalty */}
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-300">
                          Red Pressure Penalty
                        </span>
                        <span className="font-semibold text-orange-300 font-mono">
                          -
                          {rewardMetrics?.red_pressure_penalty !== undefined
                            ? rewardMetrics.red_pressure_penalty.toFixed(2)
                            : "0.00"}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-[#030712] overflow-hidden">
                        <div
                          className="h-full bg-orange-500 rounded-full transition-all duration-300 shadow-[0_0_8px_rgba(249,115,22,0.6)]"
                          style={{
                            width: `${rewardMetrics?.red_pressure_pct ?? 0}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* Speed Bonus */}
                    <div className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-300">Avg Speed Bonus</span>
                        <span className="font-semibold text-emerald-400 font-mono">
                          +
                          {rewardMetrics?.speed_bonus !== undefined
                            ? rewardMetrics.speed_bonus.toFixed(2)
                            : "0.00"}
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-[#030712] overflow-hidden">
                        <div
                          className="h-full bg-emerald-400 rounded-full transition-all duration-300 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
                          style={{
                            width: `${rewardMetrics?.speed_pct ?? 0}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

