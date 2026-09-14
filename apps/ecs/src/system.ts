// The one system this demo has: integrate position by velocity, bounce off
// the canvas edges. Built once as an RMSL `Fn` graph, and that single graph is
// what compileJS, compileWasm and compileWGSL.compute each compile — so all
// three backends run the exact same logic, only on different hardware.
import { attribute, Fn, output, uniform, type Node } from "@random-mesh/rmsl";

export type EcsSystem = {
  program: ReturnType<typeof buildProgram> & { root: Node<"float"> };
  slots: {
    posX: string;
    posY: string;
    velX: string;
    velY: string;
    width: string;
    height: string;
    dt: string;
    outPosX: string;
    outPosY: string;
    outVelX: string;
    outVelY: string;
  };
};

function buildProgram() {
  const posX = attribute("float");
  const posY = attribute("float");
  const velX = attribute("float");
  const velY = attribute("float");
  const width = uniform("float");
  const height = uniform("float");
  const dt = uniform("float");

  const nextPosX = posX.add(velX.mul(dt));
  const nextPosY = posY.add(velY.mul(dt));
  const bouncedX = nextPosX.lessThan(0).or(nextPosX.greaterThan(width));
  const bouncedY = nextPosY.lessThan(0).or(nextPosY.greaterThan(height));

  const outPosX = output("float");
  const outPosY = output("float");
  const outVelX = output("float");
  const outVelY = output("float");

  outPosX.assign(nextPosX.clamp(0, width));
  outPosY.assign(nextPosY.clamp(0, height));
  outVelX.assign(bouncedX.select(velX.negate(), velX));
  outVelY.assign(bouncedY.select(velY.negate(), velY));

  return {
    posX,
    posY,
    velX,
    velY,
    width,
    height,
    dt,
    outPosX,
    outPosY,
    outVelX,
    outVelY,
  };
}

export function createEcsSystem(): EcsSystem {
  // `assign` is only legal inside an Fn's tracked scope, and pushes its
  // statement onto that scope rather than the output node itself — so the
  // compiled root has to be the single seq node Fn wraps around the whole
  // callback (scope + return value), not the raw output nodes, or the four
  // assigns are silently dropped. buildProgram's named-node record is
  // captured as a side effect since Fn only knows how to wrap a node return.
  let nodes!: ReturnType<typeof buildProgram>;
  const root = Fn(() => {
    nodes = buildProgram();
    return nodes.outPosX;
  })();

  return {
    program: { ...nodes, root },
    slots: {
      posX: nodes.posX.name,
      posY: nodes.posY.name,
      velX: nodes.velX.name,
      velY: nodes.velY.name,
      width: nodes.width.name,
      height: nodes.height.name,
      dt: nodes.dt.name,
      outPosX: (nodes.outPosX as any).value.slot,
      outPosY: (nodes.outPosY as any).value.slot,
      outVelX: (nodes.outVelX as any).value.slot,
      outVelY: (nodes.outVelY as any).value.slot,
    },
  };
}
