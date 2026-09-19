import { Fn, invocationIndex, storage, uniform, type Node } from "@random-mesh/rmsl";

export type EcsSystem = {
  program: { root: Node<"float"> };
  slots: {
    posX: string;
    posY: string;
    velX: string;
    velY: string;
    width: string;
    height: string;
    dt: string;
  };
};

export function createEcsSystem(): EcsSystem {
  let width!: ReturnType<typeof uniform<"float">>;
  let height!: ReturnType<typeof uniform<"float">>;
  let dt!: ReturnType<typeof uniform<"float">>;
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

    const i = invocationIndex();

    const nextPosX = posX.element(i).add(velX.element(i).mul(dt));
    const nextPosY = posY.element(i).add(velY.element(i).mul(dt));
    const bouncedX = nextPosX.lessThan(0).or(nextPosX.greaterThan(width));
    const bouncedY = nextPosY.lessThan(0).or(nextPosY.greaterThan(height));

    posX.element(i).assign(nextPosX.clamp(0, width));
    posY.element(i).assign(nextPosY.clamp(0, height));
    velX.element(i).assign(bouncedX.select(velX.element(i).negate(), velX.element(i)));
    velY.element(i).assign(bouncedY.select(velY.element(i).negate(), velY.element(i)));

    return posX.element(i);
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
    },
  };
}
