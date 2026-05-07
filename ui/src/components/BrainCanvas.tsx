import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useAxonStore } from '../store/axonStore';
import type { NeuralState } from '../store/axonStore';

// ── Region colour map ──────────────────────────────────────────────────────
const REGION_COLORS: Record<string, string> = {
  prefrontal:     '#6366f1',
  hippocampus:    '#22d3ee',
  amygdala:       '#f43f5e',
  visual:         '#a3e635',
  auditory:       '#fb923c',
  language:       '#e879f9',
  thalamus:       '#fbbf24',
  cerebellum:     '#34d399',
  association:    '#818cf8',
  default_mode:   '#94a3b8',
  social:         '#f472b6',
  metacognition:  '#c084fc',
};

// Approximate 3D positions for each region (normalised sphere coords)
const REGION_POSITIONS: Record<string, [number, number, number]> = {
  prefrontal:     [0, 0.6, 0.7],
  hippocampus:    [0.4, -0.2, 0],
  amygdala:       [0.5, -0.3, 0.2],
  visual:         [0, -0.5, -0.8],
  auditory:       [0.7, 0.1, 0],
  language:       [-0.6, 0.1, 0.3],
  thalamus:       [0, 0, 0],
  cerebellum:     [0, -0.7, -0.5],
  association:    [-0.3, 0.4, -0.3],
  default_mode:   [0, 0.2, -0.6],
  social:         [0.3, 0.5, 0.2],
  metacognition:  [-0.2, 0.7, 0.1],
};

const SPHERE_RADIUS = 2.2;
const NUM_PARTICLES = 3000;

// ── Particle system ────────────────────────────────────────────────────────
function NeuronParticles({ ns }: { ns: NeuralState }) {
  const meshRef = useRef<THREE.Points>(null!);
  const geo = useRef<THREE.BufferGeometry>(null!);

  // Build initial positions biased towards region centres
  const { positions, colors, regionIdx } = useMemo(() => {
    const positions = new Float32Array(NUM_PARTICLES * 3);
    const colors    = new Float32Array(NUM_PARTICLES * 3);
    const regionIdx = new Int32Array(NUM_PARTICLES);

    const regionKeys = Object.keys(REGION_POSITIONS);
    for (let i = 0; i < NUM_PARTICLES; i++) {
      const rk = regionKeys[i % regionKeys.length];
      const [rx, ry, rz] = REGION_POSITIONS[rk];
      // place on sphere surface biased towards region
      const theta = Math.acos(2 * Math.random() - 1);
      const phi   = Math.random() * Math.PI * 2;
      const sx = Math.sin(theta) * Math.cos(phi);
      const sy = Math.sin(theta) * Math.sin(phi);
      const sz = Math.cos(theta);
      // lerp between random sphere point and region centre
      const t = 0.4 + Math.random() * 0.4;
      positions[i * 3]     = (sx * (1 - t) + rx * t) * SPHERE_RADIUS;
      positions[i * 3 + 1] = (sy * (1 - t) + ry * t) * SPHERE_RADIUS;
      positions[i * 3 + 2] = (sz * (1 - t) + rz * t) * SPHERE_RADIUS;
      const col = new THREE.Color(REGION_COLORS[rk] ?? '#ffffff');
      colors[i * 3]     = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;
      regionIdx[i] = regionKeys.indexOf(rk);
    }
    return { positions, colors, regionIdx };
  }, []);

  // Store activation per-region for animation
  const activationRef = useRef<Record<string, number>>({});

  useEffect(() => {
    if (ns.regions) activationRef.current = ns.regions;
  }, [ns.regions]);

  // Animation loop: pulse brightness and size by region activation
  const t = useRef(0);
  useFrame((_, delta) => {
    t.current += delta;
    if (!meshRef.current) return;
    const geo = meshRef.current.geometry;
    const col = geo.attributes.color as THREE.BufferAttribute;
    const regionKeys = Object.keys(REGION_POSITIONS);

    for (let i = 0; i < NUM_PARTICLES; i++) {
      const rk  = regionKeys[regionIdx[i] % regionKeys.length];
      const act = activationRef.current[rk] ?? 0.1;
      const base = new THREE.Color(REGION_COLORS[rk] ?? '#ffffff');
      // Pulse: brightness oscillates at ~1–3Hz per region
      const pulse = 0.4 + act * 0.6 + Math.sin(t.current * (1 + act * 3) + i * 0.01) * 0.2 * act;
      col.setXYZ(i, base.r * pulse, base.g * pulse, base.b * pulse);
    }
    col.needsUpdate = true;
    meshRef.current.rotation.y += delta * 0.03;
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry ref={geo}
        onUpdate={(self) => {
          self.setAttribute('position', new THREE.BufferAttribute(positions, 3));
          self.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
        }}
      />
      <pointsMaterial size={0.045} vertexColors sizeAttenuation transparent opacity={0.85} />
    </points>
  );
}

// ── Synapse arcs (top_routes) ──────────────────────────────────────────────
function SynapseArcs({ ns }: { ns: NeuralState }) {
  const linesRef = useRef<THREE.Group>(null!);

  const lines = useMemo(() => {
    const routes = ns.top_routes ?? [];
    return routes.slice(0, 8).map((r: any, i: number) => {
      const src = REGION_POSITIONS[r.src_region ?? r.src] ?? [0, 0, 0];
      const dst = REGION_POSITIONS[r.dst_region ?? r.dst] ?? [0, 0, 0];
      const mid: [number, number, number] = [
        (src[0] + dst[0]) / 2,
        (src[1] + dst[1]) / 2 + 0.6,
        (src[2] + dst[2]) / 2,
      ];
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(...src).multiplyScalar(SPHERE_RADIUS * 0.85),
        new THREE.Vector3(...mid).multiplyScalar(SPHERE_RADIUS * 0.85),
        new THREE.Vector3(...dst).multiplyScalar(SPHERE_RADIUS * 0.85),
      );
      const pts = curve.getPoints(40);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const w = Math.min(1, (r.weight ?? 0.3) * 2);
      const col = new THREE.Color(REGION_COLORS[r.src_region ?? r.src] ?? '#6366f1');
      const mat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: w * 0.6 });
      return { geo, mat, key: i };
    });
  }, [ns.top_routes]);

  return (
    <group ref={linesRef}>
      {lines.map(({ geo, mat, key }) => (
        <primitive key={key} object={new THREE.Line(geo, mat)} />
      ))}
    </group>
  );
}

// ── Region orbs ────────────────────────────────────────────────────────────
function RegionOrbs({ ns }: { ns: NeuralState }) {
  const groupRef = useRef<THREE.Group>(null!);
  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    if (!groupRef.current) return;
    groupRef.current.children.forEach((child, i) => {
      const mesh = child as THREE.Mesh;
      const rk = Object.keys(REGION_POSITIONS)[i];
      const act = (ns.regions ?? {})[rk] ?? 0.05;
      const scale = 0.06 + act * 0.12 + Math.sin(t.current * 2 + i) * 0.01 * act;
      mesh.scale.setScalar(scale);
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.3 + act * 1.2;
    });
  });

  return (
    <group ref={groupRef}>
      {Object.entries(REGION_POSITIONS).map(([rk, pos]) => {
        const col = new THREE.Color(REGION_COLORS[rk] ?? '#ffffff');
        return (
          <mesh key={rk} position={pos.map((v) => v * SPHERE_RADIUS * 0.85) as [number, number, number]}>
            <sphereGeometry args={[1, 12, 12]} />
            <meshStandardMaterial
              color={col}
              emissive={col}
              emissiveIntensity={0.5}
              transparent
              opacity={0.7}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ── Outer glass sphere ─────────────────────────────────────────────────────
function GlassSphere() {
  return (
    <mesh>
      <sphereGeometry args={[SPHERE_RADIUS + 0.05, 48, 48]} />
      <meshStandardMaterial
        color="#6366f1"
        transparent
        opacity={0.04}
        wireframe={false}
        side={THREE.BackSide}
      />
    </mesh>
  );
}

function WireFrame() {
  return (
    <mesh>
      <sphereGeometry args={[SPHERE_RADIUS + 0.06, 16, 12]} />
      <meshBasicMaterial color="#1e1b4b" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

// ── Scene ──────────────────────────────────────────────────────────────────
function Scene() {
  const ns = useAxonStore((s) => s.neuralState);
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={1.5} color="#6366f1" />
      <pointLight position={[-5, -3, -5]} intensity={0.8} color="#22d3ee" />
      <NeuronParticles ns={ns} />
      <SynapseArcs ns={ns} />
      <RegionOrbs ns={ns} />
      <GlassSphere />
      <WireFrame />
      <OrbitControls enablePan={false} minDistance={3} maxDistance={8} autoRotate autoRotateSpeed={0.4} />
    </>
  );
}

// ── Exported component ─────────────────────────────────────────────────────
export default function BrainCanvas() {
  return (
    <div style={{ width: '100%', height: '100%', background: '#020205' }}>
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 55 }}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#020205']} />
        <Scene />
      </Canvas>
    </div>
  );
}
