import { Fn, invocationIndex, storage, uniform, type Node } from "@random-mesh/rmsl";

export type ForceSystem = {
  program: { root: Node<"void"> };
  slots: {
    posX: string;
    posY: string;
    velX: string;
    velY: string;
    pointerX: string;
    pointerY: string;
    mode: string;
    strength: string;
    swirl: string;
    dt: string;
  };
};

/**
 * Pointer attractor/repulsor: reads position + a pointer uniform, writes
 * velocity. Runs before {@link createIntegrationSystem} each frame — the
 * two are dispatched as one array of roots, so this system's velocity
 * write and integration's velocity read land in the same invocation.
 */
export function createForceSystem(): ForceSystem {
  let posX!: ReturnType<typeof storage<"float">>;
  let posY!: ReturnType<typeof storage<"float">>;
  let velX!: ReturnType<typeof storage<"float">>;
  let velY!: ReturnType<typeof storage<"float">>;
  let pointerX!: ReturnType<typeof uniform<"float">>;
  let pointerY!: ReturnType<typeof uniform<"float">>;
  let mode!: ReturnType<typeof uniform<"float">>;
  let strength!: ReturnType<typeof uniform<"float">>;
  let swirl!: ReturnType<typeof uniform<"float">>;
  let dt!: ReturnType<typeof uniform<"float">>;

  const root = Fn(() => {
    posX = storage("posX", "float", { access: "read_write" });
    posY = storage("posY", "float", { access: "read_write" });
    velX = storage("velX", "float", { access: "read_write" });
    velY = storage("velY", "float", { access: "read_write" });
    pointerX = uniform("float");
    pointerY = uniform("float");
    mode = uniform("float");
    strength = uniform("float");
    swirl = uniform("float");
    dt = uniform("float");

    const i = invocationIndex();

    const dx = pointerX.sub(posX.element(i));
    const dy = pointerY.sub(posY.element(i));
    // Floored well above zero, not just offset — an inverse-square pull
    // with no damping (see createIntegrationSystem) overshoots and flips
    // sign every frame once accel gets too large at close range.
    const distSq = dx.mul(dx).add(dy.mul(dy)).max(2500);
    const radialX = dx.mul(strength).mul(mode).div(distSq);
    const radialY = dy.mul(strength).mul(mode).div(distSq);
    // A purely radial pull drives every particle onto the same point with
    // the same velocity, so releasing the pointer leaves one frozen clump
    // instead of a gas. This tangential term (perpendicular to the radial
    // one) gives each particle its own orbit around the pointer instead.
    // Scaled by |mode| rather than mode itself — its handedness shouldn't
    // flip between attract/repel, only turn off when the pointer is inactive
    // (mode 0), same as the radial term, or particles never stop orbiting.
    const active = mode.abs();
    const swirlX = dy.negate().mul(swirl).mul(active).div(distSq);
    const swirlY = dx.mul(swirl).mul(active).div(distSq);

    const accelX = radialX.add(swirlX);
    const accelY = radialY.add(swirlY);

    velX.element(i).assign(velX.element(i).add(accelX.mul(dt)));
    velY.element(i).assign(velY.element(i).add(accelY.mul(dt)));
  })();

  return {
    program: { root },
    slots: {
      posX: posX.name,
      posY: posY.name,
      velX: velX.name,
      velY: velY.name,
      pointerX: pointerX.name,
      pointerY: pointerY.name,
      mode: mode.name,
      strength: strength.name,
      swirl: swirl.name,
      dt: dt.name,
    },
  };
}

export type IntegrationSystem = {
  program: { root: Node<"void"> };
  slots: {
    posX: string;
    posY: string;
    velX: string;
    velY: string;
    width: string;
    height: string;
    dt: string;
    damping: string;
  };
};

export function createIntegrationSystem(): IntegrationSystem {
  let width!: ReturnType<typeof uniform<"float">>;
  let height!: ReturnType<typeof uniform<"float">>;
  let dt!: ReturnType<typeof uniform<"float">>;
  let damping!: ReturnType<typeof uniform<"float">>;
  let posX!: ReturnType<typeof storage<"float">>;
  let posY!: ReturnType<typeof storage<"float">>;
  let velX!: ReturnType<typeof storage<"float">>;
  let velY!: ReturnType<typeof storage<"float">>;

  const root = Fn(() => {
    posX = storage("posX", "float", { access: "read_write" });
    posY = storage("posY", "float", { access: "read_write" });
    velX = storage("velX", "float", { access: "read_write" });
    velY = storage("velY", "float", { access: "read_write" });
    width = uniform("float");
    height = uniform("float");
    dt = uniform("float");
    damping = uniform("float");

    const i = invocationIndex();

    const nextPosX = posX.element(i).add(velX.element(i).mul(dt));
    const nextPosY = posY.element(i).add(velY.element(i).mul(dt));
    const bouncedX = nextPosX.lessThan(0).or(nextPosX.greaterThan(width));
    const bouncedY = nextPosY.lessThan(0).or(nextPosY.greaterThan(height));

    posX.element(i).assign(nextPosX.clamp(0, width));
    posY.element(i).assign(nextPosY.clamp(0, height));
    // `damping` is 1 (a no-op) whenever the pointer force is inactive, so
    // particles keep their existing velocity and float freely — it only
    // drops below 1 while a sustained pull needs somewhere to bleed off,
    // otherwise held-in-place energy would compound frame over frame.
    velX.element(i).assign(bouncedX.select(velX.element(i).negate(), velX.element(i)).mul(damping));
    velY.element(i).assign(bouncedY.select(velY.element(i).negate(), velY.element(i)).mul(damping));
  })();

  return {
    program: { root },
    slots: {
      posX: posX.name,
      posY: posY.name,
      velX: velX.name,
      velY: velY.name,
      width: width.name,
      height: height.name,
      dt: dt.name,
      damping: damping.name,
    },
  };
}
