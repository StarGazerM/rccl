use proc_macro2::Span;
use std::collections::{BTreeMap, BTreeSet};
use syn::{
    Ident, LitInt, Result, Token, braced, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

mod keyword {
    syn::custom_keyword!(targets);
    syn::custom_keyword!(support);
    syn::custom_keyword!(schedule);
    syn::custom_keyword!(primitive);
    syn::custom_keyword!(target);
}

#[derive(Debug)]
pub(crate) struct CubProgram {
    pub(crate) targets: Vec<Ident>,
    pub(crate) supports: Vec<Support>,
    pub(crate) schedules: Vec<Schedule>,
    pub(crate) primitives: Vec<Primitive>,
}

#[derive(Debug)]
pub(crate) struct Support {
    pub(crate) name: Ident,
    pub(crate) fields: BTreeMap<String, Ident>,
}

#[derive(Debug)]
pub(crate) struct Schedule {
    pub(crate) name: Ident,
    pub(crate) targets: Vec<TargetSchedule>,
}

#[derive(Debug)]
pub(crate) struct TargetSchedule {
    pub(crate) target: Ident,
    pub(crate) fields: BTreeMap<String, LitInt>,
}

#[derive(Debug)]
pub(crate) struct Primitive {
    pub(crate) name: Ident,
    pub(crate) status: Ident,
    pub(crate) scope: Ident,
    pub(crate) requires: Vec<Ident>,
    pub(crate) schedule: Option<Ident>,
    pub(crate) kernels: Vec<Ident>,
}

impl Parse for CubProgram {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::targets>()?;
        let targets = parse_ident_list(input)?;
        input.parse::<Token![;]>()?;

        let mut supports = Vec::new();
        let mut schedules = Vec::new();
        let mut primitives = Vec::new();
        while !input.is_empty() {
            if input.peek(keyword::support) {
                supports.push(input.parse()?);
            } else if input.peek(keyword::schedule) {
                schedules.push(input.parse()?);
            } else if input.peek(keyword::primitive) {
                primitives.push(input.parse()?);
            } else {
                return Err(input.error("expected `support`, `schedule`, or `primitive`"));
            }
        }
        Ok(Self {
            targets,
            supports,
            schedules,
            primitives,
        })
    }
}

impl Parse for Support {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::support>()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);
        let mut fields = BTreeMap::new();
        while !content.is_empty() {
            let key: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            let value: Ident = content.parse()?;
            content.parse::<Token![;]>()?;
            insert_unique(&mut fields, key, value)?;
        }
        Ok(Self { name, fields })
    }
}

impl Parse for Schedule {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::schedule>()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);
        let mut targets = Vec::new();
        while !content.is_empty() {
            targets.push(content.parse()?);
        }
        Ok(Self { name, targets })
    }
}

impl Parse for TargetSchedule {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::target>()?;
        let target = input.parse()?;
        let content;
        braced!(content in input);
        let mut fields = BTreeMap::new();
        while !content.is_empty() {
            let key: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            let value: LitInt = content.parse()?;
            content.parse::<Token![;]>()?;
            insert_unique(&mut fields, key, value)?;
        }
        Ok(Self { target, fields })
    }
}

impl Parse for Primitive {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::primitive>()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);

        let mut status = None;
        let mut scope = None;
        let mut requires = None;
        let mut schedule = None;
        let mut kernels = None;
        while !content.is_empty() {
            let field: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            match field.to_string().as_str() {
                "status" => set_once(&mut status, content.parse::<Ident>()?, &field)?,
                "scope" => set_once(&mut scope, content.parse::<Ident>()?, &field)?,
                "requires" => set_once(&mut requires, parse_ident_list(&content)?, &field)?,
                "schedule" => {
                    let value: Ident = content.parse()?;
                    let value = (value != "none").then_some(value);
                    set_once(&mut schedule, value, &field)?;
                }
                "kernels" => set_once(&mut kernels, parse_ident_list(&content)?, &field)?,
                _ => return Err(syn::Error::new(field.span(), "unknown primitive field")),
            }
            content.parse::<Token![;]>()?;
        }

        Ok(Self {
            name,
            status: required(status, "status", input.span())?,
            scope: required(scope, "scope", input.span())?,
            requires: required(requires, "requires", input.span())?,
            schedule: required(schedule, "schedule", input.span())?,
            kernels: required(kernels, "kernels", input.span())?,
        })
    }
}

fn parse_ident_list(input: ParseStream<'_>) -> Result<Vec<Ident>> {
    let content;
    bracketed!(content in input);
    Ok(Punctuated::<Ident, Token![,]>::parse_terminated(&content)?
        .into_iter()
        .collect())
}

fn insert_unique<T>(map: &mut BTreeMap<String, T>, key: Ident, value: T) -> Result<()> {
    let name = key.to_string();
    if map.insert(name.clone(), value).is_some() {
        return Err(syn::Error::new(
            key.span(),
            format!("duplicate field `{name}`"),
        ));
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, field: &Ident) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(syn::Error::new(
            field.span(),
            format!("duplicate `{field}` field"),
        ));
    }
    Ok(())
}

fn required<T>(value: Option<T>, field: &str, span: Span) -> Result<T> {
    value.ok_or_else(|| syn::Error::new(span, format!("missing `{field}` field")))
}

impl CubProgram {
    pub(crate) fn validate(&self) -> Result<()> {
        let targets = unique_names("target", self.targets.iter())?;
        if targets.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "target set must not be empty",
            ));
        }
        unique_names("support", self.supports.iter().map(|item| &item.name))?;
        let schedules = unique_names("schedule", self.schedules.iter().map(|item| &item.name))?;
        let primitives = unique_names("primitive", self.primitives.iter().map(|item| &item.name))?;

        let allowed_status = ["planned", "implemented", "native_rust", "backend_blocked"];
        let allowed_scope = [
            "thread", "warp", "block", "agent", "grid", "device", "iterator",
        ];
        for support in &self.supports {
            for required_field in ["rust", "ownership", "capability"] {
                if !support.fields.contains_key(required_field) {
                    return Err(syn::Error::new(
                        support.name.span(),
                        format!("support `{}` is missing `{required_field}`", support.name),
                    ));
                }
            }
        }
        for schedule in &self.schedules {
            let rows = unique_names(
                "schedule target",
                schedule.targets.iter().map(|row| &row.target),
            )?;
            if rows != targets {
                return Err(syn::Error::new(
                    schedule.name.span(),
                    format!(
                        "schedule `{}` targets do not match the program target set",
                        schedule.name
                    ),
                ));
            }
            let first_fields = schedule
                .targets
                .first()
                .map(|row| row.fields.keys().cloned().collect::<BTreeSet<_>>())
                .unwrap_or_default();
            if first_fields.is_empty() {
                return Err(syn::Error::new(
                    schedule.name.span(),
                    "schedule rows must contain literal fields",
                ));
            }
            for row in &schedule.targets {
                let fields = row.fields.keys().cloned().collect::<BTreeSet<_>>();
                if fields != first_fields {
                    return Err(syn::Error::new(
                        row.target.span(),
                        format!(
                            "schedule `{}` has inconsistent target fields",
                            schedule.name
                        ),
                    ));
                }
            }
        }
        for primitive in &self.primitives {
            if !allowed_status.contains(&primitive.status.to_string().as_str()) {
                return Err(syn::Error::new(
                    primitive.status.span(),
                    "unknown primitive status",
                ));
            }
            if !allowed_scope.contains(&primitive.scope.to_string().as_str()) {
                return Err(syn::Error::new(
                    primitive.scope.span(),
                    "unknown primitive scope",
                ));
            }
            for dependency in &primitive.requires {
                if !primitives.contains(&dependency.to_string()) {
                    return Err(syn::Error::new(
                        dependency.span(),
                        format!("unknown primitive dependency `{dependency}`"),
                    ));
                }
            }
            if let Some(schedule) = &primitive.schedule
                && !schedules.contains(&schedule.to_string())
            {
                return Err(syn::Error::new(
                    schedule.span(),
                    format!("unknown schedule `{schedule}`"),
                ));
            }
            unique_names("kernel", primitive.kernels.iter())?;
        }
        reject_dependency_cycles(&self.primitives)
    }
}

fn unique_names<'a>(
    kind: &str,
    names: impl IntoIterator<Item = &'a Ident>,
) -> Result<BTreeSet<String>> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name.to_string()) {
            return Err(syn::Error::new(
                name.span(),
                format!("duplicate {kind} `{name}`"),
            ));
        }
    }
    Ok(unique)
}

fn reject_dependency_cycles(primitives: &[Primitive]) -> Result<()> {
    let by_name = primitives
        .iter()
        .map(|primitive| (primitive.name.to_string(), primitive))
        .collect::<BTreeMap<_, _>>();
    let mut done = BTreeSet::new();
    let mut active = Vec::<String>::new();

    fn visit(
        name: &str,
        by_name: &BTreeMap<String, &Primitive>,
        done: &mut BTreeSet<String>,
        active: &mut Vec<String>,
    ) -> Result<()> {
        if done.contains(name) {
            return Ok(());
        }
        if let Some(position) = active.iter().position(|candidate| candidate == name) {
            let mut cycle = active[position..].to_vec();
            cycle.push(name.to_owned());
            return Err(syn::Error::new(
                by_name[name].name.span(),
                format!("primitive dependency cycle: {}", cycle.join(" -> ")),
            ));
        }
        active.push(name.to_owned());
        for dependency in &by_name[name].requires {
            visit(&dependency.to_string(), by_name, done, active)?;
        }
        active.pop();
        done.insert(name.to_owned());
        Ok(())
    }

    for name in by_name.keys() {
        visit(name, &by_name, &mut done, &mut active)?;
    }
    Ok(())
}
