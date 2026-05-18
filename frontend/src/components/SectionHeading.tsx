interface SectionHeadingProps {
  /** Two-digit section index, e.g. "1" renders as "01". */
  readonly index: string;
  readonly kicker: string;
  readonly title: string;
}

/** The numbered "01 / KICKER — Title" heading used to open every section. */
export function SectionHeading({ index, kicker, title }: SectionHeadingProps) {
  return (
    <div className="heading">
      <span className="heading__index" aria-hidden="true">
        {index}
      </span>
      <span className="heading__text">
        <span className="heading__kicker">{kicker}</span>
        <h2 className="heading__title">{title}</h2>
      </span>
    </div>
  );
}
